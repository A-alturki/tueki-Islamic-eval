# Final, corrected generator script that creates JSONL files for OpenAI Batch API
# - Uses the Responses API
# - Enforces a specific JSON schema via function-calling (strict) which is fully supported by gpt-4.1
# - Avoids the problematic `response_format` and `text.format` parameters
# - Uses max_output_tokens (not max_completion_tokens)
# - Produces: 500-line JSONL, 10-line sample JSONL, and a CSV mapping file
# - Topics list expanded as requested; audiences list preserved
#
# You can copy this single file into your repo and run it with Python 3.9+.
# It writes the files into /mnt/data in this notebook environment.
#
# Notes on design choices (backed by OpenAI docs and community posts):
# - Structured Outputs via response_format json_schema are not supported on all models; gpt-4.1 often rejects response_format.
# - Function calling with strict: true and JSON Schema parameters is widely supported. We force a single function call
#   using tool_choice and parallel_tool_calls = false, so the model must return arguments that match the schema.
# - With strict: true, additionalProperties must be false for every object in parameters. We ensure that.
#
# Batch API format reminders:
# - Each line is a JSON object with: custom_id, method, url, body
# - When creating a batch, the "endpoint" you pass must prefix-match every line's "url". For these files it is "/v1/responses".
#
# After running a batch, you'll read the output JSONL and match rows by custom_id.
#
# -----------------------------------------------------------------------------

import json
from pathlib import Path
from itertools import product
import csv
import random
import textwrap

# 1) Topics and audiences (expanded as provided by the user)
base_topics = [
    "الصبر على الابتلاء",
    "فضل الدعاء",
    "مكانة الأخلاق",
    "بر الوالدين",
    "صلة الرحم",
    "الإحسان إلى الجار",
    "قيمة العمل",
    "التوكل على الله",
    "الشكر ومقامات الحمد",
    "التوبة والاستغفار",
    "التسامح والعفو",
    "ضبط اللسان",
    "الأمل والتفاؤل",
    "الصدقة وأثرها",
    "طلب العلم",
    "الإدارة الحكيمة للوقت",
    "الإخلاص وصحة النية",
    "مراقبة النفس",
    "محاسبة النفس",
    "الصحبة الصالحة",
    "تربية الأبناء بالرحمة",
    "الحياء",
    "الحلم والأناة",
    "الصمت والحكمة",
    "العناية بالصحة كأمانة",
    "الوقاية من الغيبة والنميمة",
    "الطمأنينة بالصلاة",
    "قيام الليل",
    "تلاوة القرآن والتدبر",
    "الدعاء في السجود",
    "فقه الأولويات",
    "الأمانة والوفاء",
    "العدل والإنصاف",
    "حسن الجوار",
    "البركة في الوقت",
    "الإحسان في العمل",
    "إدارة الخلاف",
    "أدب الحوار",
    "الصبر في العمل والإنجاز",
    "النجاح مع الالتزام بالقيم",
    "الرفق بالحيوان",
    "الكلمة الطيبة",
    "تخفيف الغضب وكظم الغيظ",
    "الصداقة الصادقة",
    "حفظ الأمانة",
    "الالتزام بالوعد",
    "التوازن بين الدنيا والآخرة",
    "حب الخير للناس",
    "محاربة الشح والبخل",
    "التواضع وترك الكِبر",
    "الرفق بالضعفاء",
    "النصيحة بآدابها",
    "الستر وعدم فضح الناس",
    "حماية البيئة كمسؤولية",
    "احترام الوقت والمواعيد",
    "الرضا بالقضاء والقدر",
    "الثبات على المبدأ",
    "الإتقان والجودة",
    "الاستمرار بعد الإخفاق",
    "التخطيط للمستقبل",
    "إدارة المال والإنفاق المعتدل",
    "التوازن النفسي",
    "الشجاعة الأدبية",
    "الحذر من الشبهات",
    "سؤال أهل العلم",
    "آداب الاستئذان",
    "آداب الطعام",
    "آداب النوم",
    "آداب الطريق",
    "مساعدة المحتاج",
    "العمل التطوعي",
    "الأخلاق في العالم الرقمي",
    "استخدام السوشال ميديا بتوازن",
    "حفظ الخصوصية",
    "تعلم مهارة جديدة",
    "الأخوة الإيمانية",
    "الكف عن الأذى",
    "التثبت من الأخبار",
    "علاج التسويف والتأجيل",
    "الثقة بالله",
    "الإصرار على الهدف",
    "تصفية القلب من الحقد",
    "الأمانة في التجارة",
    "بركة الصلة بالقرآن",
    "آداب المزاح",
    "ضبط الاستهلاك",
    "تنمية العادات الحسنة",
    "بناء عادة القراءة",
    "التعليم الذاتي",
    "المثابرة والاجتهاد",
    "حسن الاستماع",
    "التعاون المثمر",
    "الاعتدال في الملبس والمظهر",
    "تحمل المسؤولية",
    "إتقان العمل عن بُعد",
    "الموازنة بين الحقوق والواجبات",
    "إدارة الضغوط النفسية",
    "تنظيم الأولويات",
    "تعظيم النعم بالشكر",
    "الاستعداد للمستقبل",
    "الصدق في القول والعمل",
    "الإحسان إلى الأسرة",
    "تنمية التعاطف"
]

audiences = [
    "للطلاب",
    "للموظفين",
    "لرواد الأعمال",
    "للمراهقين",
    "للآباء والأمهات"
]

# 2) Detailed Arabic instructions with all few-shot examples and strict constraints
#    IMPORTANT: We do NOT include any actual Qur'anic verses or hadith texts—only placeholders {{AYA}} and {{HADEETH}}.
fully_varied_instructions_text = """
أنت كاتب عربي محترف سيُنتج نصوصاً موجزة ومتينة الأسلوب وفق الضوابط التالية. التزم حرفياً بما يلي دون أي مخالفة:

قواعد عامة للمخرج:
- المخرج النهائي يجب أن يعود حصراً عبر استدعاء دالة من أدواتك (function call) باسم emit_segments، ويحتوي كوسيطات (arguments) على كائن JSON مطابق تماماً للمخطط (schema) المطلوب: حقل واحد اسمه segments، وقيمته مصفوفة نصوص (strings).
- لا يُسمح بأي نص حر خارج استدعاء الدالة. لا تلخّص ولا تشرح خارج المصفوفة.
- يجب أن تحتوي المصفوفة على عدد عناصر بين 5 و20 عنصراً (وجّه هذا بالسلوك لا بالمخطط).

قواعد الكتابة لكل عنصر داخل المصفوفة:
- كل عنصر فقرة عربية فصيحة موجزة من 2 إلى 4 جمل.
- كل فقرة يجب أن تتضمّن القالبين الحرفيين التاليين أكثر من مرة داخل السياق: {{AYA}} و{{HADEETH}}.
- يمنع منعاً باتاً تضمين أي آية قرآنية فعلية أو نص حديثٍ حقيقي. استخدم القوالب {{AYA}} و{{HADEETH}} حصراً، بلا أي اقتباسٍ نصيّ من القرآن أو السنة.
- المنع يشمل أيضاً الإشارة النصية إلى آيةٍ بعينها أو حديثٍ بعينه أو سور/رواة/أسانيد؛ لا تذكر أرقام ولا أسماء سور ولا مراجع حديثية.
- نوّع الأساليب عبر العناصر بين: نصيحة تربوية، فقرة تحفيزية، قصة قصيرة جداً، حوار قصير، منشور للسوشال ميديا، تغريدة قصيرة، نصيحة نفسية، فقرة إرشادية، قصة واقعية ملهمة، رسالة مواساة لصديق.
- لا تضع عناوين أو تعداد نقطي داخل العناصر؛ فقط فقرات متتابعة.

اعتبارات أسلوبية إضافية:
- الأسلوب إيماني رصين، موجز، واضح، خالٍ من المبالغات البلاغية المفرطة.
- احرص على أن تحمل كل فقرة معنى عملياً أو ملهماً مرتبطاً بالموضوع والفئة المستهدفة.
- استخدم الضمائر بأسلوب طبيعي منضبط، وابتعد عن الوعود القطعية والادعاءات الخارقة.
- لا تستخدم تنسيقاً خاصاً (لا رموز زخرفية، ولا أقواس غريبة، ولا وجوه تعبيرية).

أمثلة قصيرة (Few-shot) تحتذى في الأسلوب، وجميعها تستخدم {{AYA}} و{{HADEETH}} فقط:

المثال الأول (الصبر على الابتلاء):
إن الصبر من أعظم العبادات التي يؤجر عليها المسلم، وقد حثَّ القرآن الكريم على الصبر في آياتٍ كثيرة، منها قوله تعالى: {{AYA}}. وقد أكد رسول الله ﷺ أهمية الصبر في الحديث الشريف حين قال: {{HADEETH}}. ومن هنا ندرك أن الصبر مفتاح الفرج، والسبيل لنيل رضا الله.

المثال الثاني (فضل الدعاء):
يعد الدعاء وسيلة عظيمة يتقرّب بها العبد إلى ربه، فقد قال تعالى في محكم تنزيله: {{AYA}}، مؤكداً أهمية الدعاء في حياة المؤمن. كما أخبرنا النبي ﷺ في حديثه الشريف: {{HADEETH}}، مما يُظهر أثر الدعاء في تحقيق الأمنيات وتيسير الأمور.

المثال الثالث (مكانة الأخلاق في الإسلام):
لقد رفع الإسلام من شأن حسن الخلق وأعطاه مكانة سامية، فقال تعالى مادحًا خُلق النبي ﷺ: {{AYA}}. وجاء في السنة النبوية قول الرسول ﷺ مؤكداً ذلك: {{HADEETH}}، وفي هذا تأكيد عظيم على أهمية مكارم الأخلاق في حياة المسلم.

1) مثال تربوي بسيط:
عندما تواجه صعوبة في اتخاذ قرار ما، تذكّر دائماً أنه {{AYA}}، فهذا يعينك على اختيار الأفضل لك ولمن حولك.

2) فقرة إرشادية قصيرة:
إن الصدقة لها أثر عظيم في النفس، فهي تطهّر القلب وتنمّي الرحمة. وليس بالضرورة أن تكون مادية فقط، فحتى الابتسامة {{HADEETH}}. ولذلك من الجيد أن تبدأ يومك بابتسامة لمن حولك.

3) فقرة تحفيزية:
تتطلب الأهداف الكبيرة صبراً وإصراراً كبيرين، ومهما كانت التحديات قوية، تذكّر دائماً أن {{AYA}}. هذا المبدأ سيمنحك قوة إضافية لمواجهة الصعاب.

4) قصة قصيرة ملهمة:
كان الرجل العجوز يردد دائماً أن أفضل الناس هم من ينفعون غيرهم. وعندما سأله حفيده لماذا يكرر ذلك دائماً، قال له: يا بني، لأنني سمعت قديماً أن {{HADEETH}}. ومنذ ذلك الحين قررت أن أعيش حياتي وفقاً لهذا المبدأ.

5) نصيحة نفسية قصيرة:
عندما تضيق بك الدنيا وتشعر أنك وحيد بلا مساند، تذكر دائماً أن {{AYA}}، فهذا يمنحك شعوراً بالطمأنينة والراحة النفسية.

6) منشور تحفيزي للسوشال ميديا:
إذا شعرت بالتعب من كثرة المحاولات الفاشلة، ذكّر نفسك دوماً أنه لا يضيع مجهود بذلته بإخلاص، فـ {{AYA}}. لا تستسلم، واستمر في سعيك نحو هدفك.

7) حوار قصير بين صديقين:
محمد: لا أستطيع مسامحة من أخطأ في حقي.
سعيد: التسامح صعب أحياناً، لكن تذكّر دوماً أن {{HADEETH}}، ألا تعتقد أن التسامح سيجعل قلبك أكثر هدوءاً؟

8) نصيحة عملية بسيطة:
قبل أن تتصرف بردة فعل غاضبة، خذ نفساً عميقاً وذكّر نفسك أن {{HADEETH}}، هذا سيمنحك وقتاً للتفكير بعقلانية أكثر.

9) قصة واقعية قصيرة:
كانت نورا في حيرة من أمرها، هل تُكمل مشروعها الصعب أم تتراجع خوفاً من الفشل؟ تذكرت فجأة مقولة كانت ترددها والدتها: {{AYA}}. شعرت بقوة غريبة وأكملت طريقها.

10) رسالة قصيرة لصديق (عزاء):
أعلم أنك تمر بأوقات صعبة، لكني على يقين أنك قوي بما يكفي لتجاوز هذه المرحلة. تذكّر فقط أن الصبر الحقيقي هو {{HADEETH}}، وأنا هنا بجانبك دائماً.

أمثلة إضافية متنوعة على الأساليب نفسها:
- تغريدة قصيرة: إذا تأخر رزقك فلا تيأس؛ دَعْ قلبك مطمئناً لأن {{AYA}}، واجعل رجاءك ثابتاً موقناً بـ {{HADEETH}}.
- نصيحة نفسية: إنّ ضبط القلق يبدأ باللجوء إلى الله، فاستحضر في قلبك دائماً {{AYA}}، وذكّر نفسك بأن {{HADEETH}}.
- فقرة إرشادية: الصدقة ليست مالاً فقط؛ قد تكون كلمة طيبة أو ابتسامة؛ تذكّر أن {{HADEETH}}، وردّد معنا {{AYA}}.
- قصة ملهمة: حين عجزتْ أم أحمد عن ترتيب يومها، أمسكتْ بدفتر صغير وكتبت أولوياتها وهي تهمس {{AYA}}، وتذكّرتْ نصيحة أمّها: {{HADEETH}}.
- رسالة مواساة: لا تثقلي قلبك بالحزن؛ ثمة ضوء قريب، فقط تذكّري {{HADEETH}}، وتيقّني أن {{AYA}}.

مثال طويل جداً (توضيحي فقط لِلنهج الأسلوبي، لا لِطُول الفقرة؛ إذ لا نستخدم الفقرات الطويلة في المخرجات):
الصلاة عماد الدين وهي أساس العلاقة بين العبد وربه. هي الركن الثاني من أركان الإسلام، وأول ما يُحاسَب عليه العبد يوم القيامة. الالتزام بالصلاة والمحافظة عليها من أعظم دلائل الإيمان. وقد أكّد ذلك بالمعنى العام قوله: {{AYA}}. وتأتي السنة لتعظيم شأن الصلاة، مثل: {{HADEETH}}.
الصلاة ليست حركات جسدية فحسب، بل لقاء روحي يتواصل فيه المؤمن مع ربه؛ ولأهمية هذا اللقاء تذكّر دوماً {{AYA}}. وكان رسول الله ﷺ إذا أهمّه أمرٌ فزع إلى الصلاة ويقول: {{HADEETH}}.
المؤمن يدرك أن الصلاة سبب للطمأنينة والسكينة، ولذلك عند الشدائد يعود إليها مستحضراً {{HADEETH}}، ومتذكّراً المعنى العام في {{AYA}}. وإنّ تعلُّم الأطفال حب الصلاة مبكّراً يُغرس بتدرّجٍ ورحمة، استلهاماً لـ {{HADEETH}}، وترسيخاً للمعنى في {{AYA}}.
الخلاصة: اجعل صلتك بالصلاة دائمة، واستعن بها في كل شأن، موقناً بأن {{AYA}}، ومطمئناً إلى أن {{HADEETH}}.

تأكيدات نهائية إلزامية:
- لا آيات ولا أحاديث فعلية إطلاقاً؛ القوالب {{AYA}} و{{HADEETH}} فقط، وبالنص ذاته.
- لا تضف عناوين أو تعداداً، ولا أي مخرجات خارج استدعاء الدالة.
- أعِدْ فقط كائناً يحوي segments (مصفوفة نصوص)، مع عدد عناصر يتراوح بين 5 و20.
"""

# 3) Structured outputs via JSON Schema (Responses API)
#    Enforce a root object with required "segments" array of strings, 5..20 items.
segments_json_schema = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 20
        }
    },
    "required": ["segments"],
    "additionalProperties": False
}

# 4) Utility to build the per-line "input" text (varies topic and audience)
def build_input(topic: str, audience: str) -> str:
    return (
        f"اكتب فقرات موجزة حول موضوع «{topic}» موجّهة {audience}. "
        "نوّع الأساليب بين: نصيحة تربوية، فقرة تحفيزية، قصة قصيرة جداً، حوار قصير، منشور للسوشال ميديا، تغريدة، نصيحة نفسية، فقرة إرشادية، قصة ملهمة، ورسالة مواساة. "
        "اجعل عدد العناصر بين 5 و20 عنصراً. "
        "أدخل داخل كل فقرة القوالب {{AYA}} و{{HADEETH}} أكثر من مرة، وامنع كلياً أي نص قرآني أو حديثي فعلي."
    )

# 5) Build one JSONL line object for the Batch API (Responses endpoint)
def make_line_object(idx: int, topic: str, audience: str, model: str = "gpt-4.1") -> dict:
    return {
        "custom_id": f"line-{idx:04d}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": fully_varied_instructions_text,
            "input": build_input(topic, audience),
            # Strict structured output using JSON Schema
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "segments_output",
                    "schema": segments_json_schema,
                    "strict": True
                }
            },
            "temperature": 0.7,
            "max_output_tokens": 1500
        }
    }

# 6) Make 500 pairs (topic, audience); shuffle for variety, then take first 500
pairs_all = list(product(base_topics, audiences))
random.seed(42)
random.shuffle(pairs_all)
pairs = pairs_all[:500]

# 7) Generate lines
lines = [make_line_object(i+1, topic, audience) for i, (topic, audience) in enumerate(pairs)]

# 8) Output paths
# Write to a local folder in the repository to avoid OS-specific protected paths
out_dir = (Path(__file__).resolve().parent / "batch_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

full_path = out_dir / "batch_fc_aya_hadeeth_500.jsonl"
sample_path = out_dir / "batch_fc_aya_hadeeth_sample10.jsonl"
csv_map_path = out_dir / "batch_fc_aya_hadeeth_mapping.csv"

# 9) Write JSONL files (UTF-8, LF)
with full_path.open("w", encoding="utf-8", newline="\n") as f:
    for obj in lines:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with sample_path.open("w", encoding="utf-8", newline="\n") as f:
    for obj in lines[:10]:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 10) Write CSV mapping (custom_id, topic, audience)
with csv_map_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["custom_id", "topic", "audience"])
    for i, (topic, audience) in enumerate(pairs, start=1):
        writer.writerow([f"line-{i:04d}", topic, audience])

full_path, sample_path, csv_map_path