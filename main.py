import os 
import pandas as pd 
from google import genai
from pydantic  import BaseModel, Field
from dotenv import load_dotenv
from google.genai import types



load_dotenv(override=True)
api_key=os.getenv("OPENAI_KEY")
client=genai.Client(api_key=api_key)
df=pd.read_csv("scored_leads.csv")
# print(df.head())
class PersonalizedEmail(BaseModel):
    subject_line: str=Field(
        description="A compelling, professional subject line under 8 words. "
    )
    email_body:str=Field(
        description="A tailored 3 paragraph outreach email referencing similar customer success."

    )
    call_to_action:str=Field(
        description="A single low-friction CTA line asking for a 15-min chat."
    )
    print("Setup complete and Schema defined.")

generated_results = []
for index,row in df.iterrows():
    lead_name=row.get('lead_name','Valued Customer')
    company_name=row.get('company_name','your_company')
    industry=row.get('industry','your_industry')
    similar_customers=row.get('similar_customers','other successful companies')

    print(f"Processing Lead {index +1}/{len(df)}: {lead_name} at {company_name}...")
    prompt=f"""
You are an exoert in B2B Sales Outreach Agent. Write a highly personalized cold outreach email.

---Lead Details---
Name:{lead_name}
Company:{company_name}
Industry:{industry}
--- HISTORICAL SIMILAR CUSTOMERS (Context) ---
    {similar_customers}

    --- INSTRUCTIONS ---
    - Highlight how we helped a company similar to theirs achieve success.
    - Connect that specific success story to {company_name}'s probable business goals.
    - Keep the tone warm, professional, and concise. Do not use generic template filler.
    """
    try:
        response=client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PersonalizedEmail,
                temperature=0.3
            )
        )
        parsed_email=PersonalizedEmail.model_validate_json(response.text)
        generated_results.append({
            'lead_name':lead_name,
            'company_name': company_name,
            'subject_line':parsed_email.subject_line,
            "email_body": parsed_email.email_body,
            "call_to_action": parsed_email.call_to_action
        })
        print(f"Success for {company_name}")

    except Exception as e:
        print(f"❌ Error generating email for {company_name}: {e}")

results_df = pd.DataFrame(generated_results)

# Save to a new CSV file
results_df.to_csv("generated_emails.csv", index=False)

print("\n🎉 PIPELINE COMPLETE! Check 'generated_emails.csv' for the final output.")

    



