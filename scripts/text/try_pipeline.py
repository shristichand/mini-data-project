# from mini_data_project import TextDatasetPipeline, PipelineConfig

# p = TextDatasetPipeline(PipelineConfig(min_npi_score=None))

# samples = {
#     "mixed":   "<p>नेपालमा २०८१ सालमा <b>पर्यटन</b> प्रवर्द्धन भयो! Visit Nepal 2024.</p>",
#     "english": "This is an English paragraph about tourism in Nepal and nothing else.",
#     "good":    "नेपाल एक सुन्दर देश हो । यहाँ धेरै हिमाल र नदीहरू छन् र पर्यटकहरू पनि आउँछन् ।",
# }

# for name, text in samples.items():
#     r = p.process_text(text)
#     print(name, "| passed:", r.passed, "| reasons:", r.rejection_reasons)

from mini_data_project.text import (
    clean_unicode_pipeline, clean_text_pipeline, profile_text, WordTokenizer
)

# raw_text = "<p>नेपाल\u200Bमा २०८१ सालमा <b>पर्यटन</b> प्रवर्द्धन भयो! Visit Nepal 2024.</p>"
raw_text = "<div class=\"post\"><p>आजको     मौसम पूर्वानुमान</p></div> &nbsp; Copyright © 2024 सर्वाधिकार सुरक्षित | Home | About Us | Contact"

# 1. Character-level cleaning, then text-level cleaning
cleaned = clean_text_pipeline(
    clean_unicode_pipeline(raw_text),
    strip_html_tags=True,
    ascii_digits=False,   # keep Devanagari digits, as the real pipeline does
)

# 2. Tokenize and score
tokens = WordTokenizer().tokenize(cleaned)
profile = profile_text(cleaned)

# 3. Prove the invisible character was removed
print("Zero-width space in raw     :", "\u200b" in raw_text)
print("Zero-width space in cleaned :", "\u200b" in cleaned)

# 4. Write results to a file you can open with proper Devanagari rendering
with open("demo_output.txt", "w", encoding="utf-8") as f:
    f.write(f"RAW:\n{raw_text}\n\n")
    f.write(f"CLEANED:\n{cleaned}\n\n")
    f.write(f"TOKENS:\n{tokens}\n\n")
    f.write(f"QUALITY SCORE: {profile['quality_score']}\n")
    f.write(f"DEVANAGARI RATIO: {profile['deva_ratio']}\n")

print("Saved demo_output.txt")