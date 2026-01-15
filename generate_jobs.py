import csv
import random

job_roles = [
    "Frontend Engineer", "Backend Developer", "Full Stack Developer",
    "React Developer", "Java Developer", "Python Developer",
    "Data Engineer", "ML Engineer", "DevOps Engineer",
    "Software Engineer"
]

skills = [
    "React", "JavaScript", "TypeScript", "HTML", "CSS",
    "Java", "Spring Boot", "Hibernate",
    "Python", "Django", "Flask",
    "Node.js", "Express",
    "SQL", "PostgreSQL", "MongoDB",
    "Docker", "Kubernetes", "AWS", "CI/CD"
]

locations = ["remote", "onsite", "hybrid"]

link = [
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4341785224",
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4348594440", 
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4313057332"
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4358540406"
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4359373732"
]

companies = [
    "Google", "Amazon", "Microsoft", "Netflix",
    "Startup", "FinTech Company", "E-commerce Company"
]

def generate_description(role):
    selected_skills = random.sample(skills, random.randint(3, 6))
    return (
        f"We are hiring a {role}. "
        f"Required skills include {', '.join(selected_skills)}. "
        f"Experience with scalable systems and teamwork is expected."
    )

with open("jobs_big.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "title", "description", "location", "company", "url"])

    for i in range(1, 1001): 
        role = random.choice(job_roles)
        writer.writerow([
            i,
            role,
            generate_description(role),
            random.choice(locations),
            random.choice(companies),
            random.choice(link)
        ])

print("✅ jobs_big.csv generated with 50,000 records")
