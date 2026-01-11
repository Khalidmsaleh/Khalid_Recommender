# ============================================================
# Dual Recommender Synthetic Data Generator (Arabic)
# Generates:
# 1) courses.csv
# 2) trainees.csv
# 3) intake_profiles.csv
# 4) interactions.csv
# 5) role_to_tags.csv
# ============================================================

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_COURSES = 80
N_TRAINEES = 200
N_INTERACTIONS = 2000
TODAY = datetime.now()

COURSES_CSV = "courses.csv"
TRAINEES_CSV = "trainees.csv"
INTAKE_CSV = "intake_profiles.csv"
INTERACTIONS_CSV = "interactions.csv"
ROLE2TAGS_CSV = "role_to_tags.csv"

TRACKS = [
    "الجرائم المعلوماتية والتحقيقات الرقمية",
    "الأدلة الرقمية والتحليل الجنائي الرقمي",
    "الإجراءات النظامية وإدارة الأدلة",
    "الأمن السيبراني الأساسي وحماية الأنظمة",
    "مراكز العمليات الأمنية والاستجابة للحوادث",
    "تحليل البيانات لدعم التحقيقات",
    "الذكاء الاصطناعي في الأمن والتحقيقات",
    "القيادة والتشغيل وإدارة الأزمات",
]

LEVELS = ["مبتدئ", "متوسط", "متقدم"]
AUDIENCES = ["عسكري فقط", "مدني فقط", "الجميع"]

AGENCIES = [
    "النيابة العامة – وحدة الجرائم المعلوماتية",
    "الشرطة – إدارة التحريات والبحث الجنائي",
    "مختبر الأدلة الرقمية (التحليل الجنائي)",
    "إدارة الأمن السيبراني (جهة حكومية)",
    "إدارة مكافحة الاحتيال والجرائم المالية",
    "الدعم التقني للجهات العدلية والمحاكم",
    "وحدة الاستجابة للحوادث (IRT) – جهة حكومية",
    "وزارة الحرس الوطني – قطاع الأمن والحماية",
    "الحرس الملكي – الأمن والحماية",
    "القوات الجوية الملكية السعودية – الأمن السيبراني / مركز العمليات الأمنية (SOC)",
]

MILITARY_AGENCIES = {
    "وزارة الحرس الوطني – قطاع الأمن والحماية",
    "الحرس الملكي – الأمن والحماية",
    "القوات الجوية الملكية السعودية – الأمن السيبراني / مركز العمليات الأمنية (SOC)",
}

INTERESTS = [
    "جرائم معلوماتية","تصيد احتيالي","احتيال مالي","برمجيات فدية","أدلة رقمية",
    "أدلة الأجهزة الذكية","أدلة الشبكات","إجراءات نظامية وإثباتات","سلسلة حفظ الأدلة",
    "مركز عمليات أمنية","SIEM وتحليل تنبيهات","الاستجابة للحوادث","البحث عن التهديدات",
    "OSINT","تحليل بيانات","بايثون","SQL","ذكاء اصطناعي للأمن","كشف الشذوذ والأنماط",
    "تحليل برمجيات خبيثة (أساسيات)",
]

SKILLS = [
    "أساسيات التحقيق الجنائي","توثيق القضايا وكتابة المحاضر","سلسلة حفظ الأدلة",
    "الإجراءات النظامية ذات العلاقة","كتابة التقارير المهنية","التعامل مع الأدلة الرقمية",
    "تحليل أجهزة التخزين","التحليل الجنائي للذاكرة","التحليل الجنائي للأجهزة الذكية",
    "التحليل الجنائي للشبكات","أساسيات مركز العمليات الأمنية (SOC)","تحليل السجلات",
    "أساسيات أنظمة SIEM","الاستجابة للحوادث السيبرانية","إدارة الثغرات الأمنية",
    "البحث عن التهديدات","تحليل البيانات (Excel/مبادئ)","أساسيات بايثون للتحليل",
    "أساسيات SQL","OSINT (استخبارات المصادر المفتوحة)","أساسيات تحليل البرمجيات الخبيثة",
    "التواصل والعرض","التخطيط التشغيلي","إدارة الأزمات","أخلاقيات العمل والامتثال","القيادة الأساسية",
]

GOALS = [
    "تطوير مهارات التحقيق","التخصص في الأدلة الرقمية","الاستعداد للعمل في مركز عمليات أمنية",
    "تحسين كتابة التقارير والمخرجات الرسمية","ترقية وظيفية/رفع جاهزية الأداء",
    "الانتقال لمسار الأمن السيبراني","تعلم تحليل البيانات لدعم القضايا",
    "فهم تطبيقات الذكاء الاصطناعي في العمل","رفع الوعي بالامتثال وأخلاقيات التقنية",
]

ROLES = [
    "محقق جرائم معلوماتية","محلل أدلة رقمية","محلل مركز عمليات أمنية (SOC)",
    "باحث قانوني/إجرائي","محلل بيانات للتحقيقات","مختص أمن سيبراني",
    "ضابط عمليات وأمن","مختص جرائم مالية","مختص دعم تقني قضائي",
]

def pick_k(lst, kmin=2, kmax=4):
    k = random.randint(kmin, kmax)
    return random.sample(lst, k)

def trainee_id(i):
    return f"متدرب-{i:03d}"

# -----------------------------
# role_to_tags.csv
# -----------------------------
role_tags = {
    "محقق جرائم معلوماتية": "جرائم معلوماتية,إجراءات نظامية وإثباتات,OSINT,تحليل بيانات",
    "محلل أدلة رقمية": "أدلة رقمية,أدلة الأجهزة الذكية,أدلة الشبكات,سلسلة حفظ الأدلة",
    "محلل مركز عمليات أمنية (SOC)": "مركز عمليات أمنية,SIEM وتحليل تنبيهات,الاستجابة للحوادث,البحث عن التهديدات",
    "باحث قانوني/إجرائي": "إجراءات نظامية وإثباتات,سلسلة حفظ الأدلة,كتابة تقارير",
    "محلل بيانات للتحقيقات": "تحليل بيانات,SQL,بايثون,كشف الشذوذ والأنماط",
    "مختص أمن سيبراني": "الاستجابة للحوادث,إدارة الثغرات الأمنية,البحث عن التهديدات,برمجيات فدية",
    "ضابط عمليات وأمن": "القيادة والتشغيل وإدارة الأزمات,إدارة الأزمات,التخطيط التشغيلي",
    "مختص جرائم مالية": "احتيال مالي,تصيد احتيالي,تحليل بيانات,إجراءات نظامية وإثباتات",
    "مختص دعم تقني قضائي": "أدلة رقمية,كتابة تقارير,التعامل مع الأدلة الرقمية",
}

df_role2tags = pd.DataFrame([{"role": k, "tags": v} for k, v in role_tags.items()])
df_role2tags.to_csv(ROLE2TAGS_CSV, index=False, encoding="utf-8-sig")

# -----------------------------
# courses.csv
# -----------------------------
courses = []
for i in range(1, N_COURSES + 1):
    cid = f"C{i:03d}"
    track = random.choice(TRACKS)
    level = random.choice(LEVELS)
    audience = random.choice(AUDIENCES)
    kws = pick_k(INTERESTS, 3, 5)
    title = f"دورة {track} ({level}) #{i}"
    desc = f"هذه الدورة تغطي: {', '.join(kws)}. مناسبة لتطوير مهارات المشاركين في {track} مع تطبيقات عملية."
    courses.append({
        "course_id": cid,
        "title": title,
        "track": track,
        "level": level,
        "audience": audience,
        "description": desc
    })

pd.DataFrame(courses).to_csv(COURSES_CSV, index=False, encoding="utf-8-sig")

# -----------------------------
# trainees.csv
# -----------------------------
trainees = []
for i in range(1, N_TRAINEES + 1):
    tid = trainee_id(i)
    agency = random.choice(AGENCIES)
    military = "نعم" if agency in MILITARY_AGENCIES else "لا"
    role = random.choice(ROLES)
    trainees.append({
        "trainee_id": tid,
        "agency": agency,
        "military": military,
        "role": role
    })

pd.DataFrame(trainees).to_csv(TRAINEES_CSV, index=False, encoding="utf-8-sig")

# -----------------------------
# intake_profiles.csv
# -----------------------------
intake_rows = []
for i in range(1, N_TRAINEES + 1):
    tid = trainee_id(i)

    if random.random() < 0.15:
        interests = ""
        skills = ""
        goals = ""
        notes = ""
    else:
        interests = ", ".join(pick_k(INTERESTS, 2, 5))
        skills = ", ".join(pick_k(SKILLS, 2, 5))
        goals = random.choice(GOALS)
        notes = random.choice([
            "يرغب في تطبيق عملي وحالات واقعية.",
            "يحتاج مسار تدريجي من مبتدئ إلى متقدم.",
            "يركز على تحسين كتابة التقارير الرسمية.",
            "مهتم بتعلم أدوات التحليل الرقمي.",
            "يريد فهم آليات الاستجابة للحوادث."
        ])

    intake_rows.append({
        "trainee_id": tid,
        "interests": interests,
        "skills": skills,
        "goals": goals,
        "notes": notes
    })

pd.DataFrame(intake_rows).to_csv(INTAKE_CSV, index=False, encoding="utf-8-sig")

# -----------------------------
# interactions.csv
# -----------------------------
interactions = []
course_ids = [f"C{i:03d}" for i in range(1, N_COURSES + 1)]

for _ in range(N_INTERACTIONS):
    tid = trainee_id(random.randint(1, N_TRAINEES))
    cid = random.choice(course_ids)
    event = random.choice(["enroll", "complete"])
    days_ago = random.randint(1, 180)
    ts = (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    interactions.append({
        "trainee_id": tid,
        "course_id": cid,
        "event": event,
        "timestamp": ts
    })

pd.DataFrame(interactions).to_csv(INTERACTIONS_CSV, index=False, encoding="utf-8-sig")

print("✅ تم إنشاء ملفات البيانات بنجاح:")
print(COURSES_CSV, TRAINEES_CSV, INTAKE_CSV, INTERACTIONS_CSV, ROLE2TAGS_CSV)
