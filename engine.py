import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm_id(raw_id: str) -> str:
    s = str(raw_id).strip()
    s = s.replace("Trainee-", "").replace("trainee-", "")
    s = s.replace("متدرب-", "").replace("متدرّب-", "")
    s = re.sub(r"\D+", "", s)
    if not s:
        raise ValueError("Invalid trainee_id")
    return f"متدرب-{int(s):03d}"


def safe_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def join_text(*parts) -> str:
    return " ".join([p for p in (safe_str(x) for x in parts) if p])


class RecommenderEngine:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.load_data()
        self.build_models()

    def load_data(self):
        self.courses = pd.read_csv(f"{self.data_path}/courses.csv")
        self.trainees = pd.read_csv(f"{self.data_path}/trainees.csv")
        self.intake = pd.read_csv(f"{self.data_path}/intake_profiles.csv")
        self.interactions = pd.read_csv(f"{self.data_path}/interactions.csv")
        self.role_to_tags = pd.read_csv(f"{self.data_path}/role_to_tags.csv")

        self.trainees["trainee_id"] = self.trainees["trainee_id"].apply(norm_id)
        self.intake["trainee_id"] = self.intake["trainee_id"].apply(norm_id)
        self.interactions["trainee_id"] = self.interactions["trainee_id"].apply(norm_id)

        self.courses["course_text"] = self.courses.apply(
            lambda r: join_text(r.get("title",""), r.get("track",""), r.get("level",""),
                                r.get("audience",""), r.get("description","")),
            axis=1
        )

        self.role_tags_map = {
            safe_str(r["role"]): safe_str(r["tags"])
            for _, r in self.role_to_tags.iterrows()
        }

        self.intake_map = self.intake.set_index("trainee_id").to_dict(orient="index")

    def build_models(self):
        self.vectorizer = TfidfVectorizer()
        self.course_vectors = self.vectorizer.fit_transform(self.courses["course_text"])

    def build_profile_text(self, trainee_id):
        trainee = self.trainees[self.trainees["trainee_id"] == trainee_id].iloc[0]
        intake = self.intake_map.get(trainee_id, {})

        tags = self.role_tags_map.get(trainee["role"], "")

        return join_text(
            trainee["agency"],
            trainee["role"],
            intake.get("interests",""),
            intake.get("skills",""),
            intake.get("goals",""),
            intake.get("notes",""),
            tags
        )

    def recommend_for_trainee(self, trainee_id, top_k=10):
        trainee_id = norm_id(trainee_id)
        profile_text = self.build_profile_text(trainee_id)

        q = self.vectorizer.transform([profile_text])
        sims = cosine_similarity(q, self.course_vectors).flatten()

        idx = np.argsort(-sims)[:top_k]

        results = []
        for i in idx:
            c = self.courses.iloc[i]
            results.append({
                "course_id": c["course_id"],
                "title": c["title"],
                "track": c["track"],
                "level": c["level"],
                "score": float(sims[i])
            })

        return results

    def recommend_for_course(self, course_id, top_k=10):
        course = self.courses[self.courses["course_id"] == course_id].iloc[0]
        q = self.vectorizer.transform([course["course_text"]])

        profiles = []
        for tid in self.trainees["trainee_id"]:
            text = self.build_profile_text(tid)
            profiles.append(text)

        t_vecs = self.vectorizer.transform(profiles)
        sims = cosine_similarity(q, t_vecs).flatten()

        idx = np.argsort(-sims)[:top_k]

        results = []
        for i in idx:
            t = self.trainees.iloc[i]
            results.append({
                "trainee_id": t["trainee_id"],
                "agency": t["agency"],
                "role": t["role"],
                "score": float(sims[i])
            })

        return results
