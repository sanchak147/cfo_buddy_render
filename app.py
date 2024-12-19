from flask import Flask, request, render_template, jsonify, send_file
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from pandasai.llm.openai import OpenAI
from PyPDF2 import PdfReader
from io import StringIO
import re
from pandasai import Agent

# Load environment variables
# load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize ChatGroq and PandasAI
chat_groq = ChatGroq()
# Define JSON schema for structured financial analysis
json_schema = {
    "title": "Financial Analysis",
    "description": "Detailed financial analysis including insights and comparisons.",
    "type": "object",
    "properties": {
        "text1": {
            "type": "string",
            "description": "Unstructured insights regarding the financial status."
        },
        "table": {
            "type": "object",
            "description": "Financial comparison table over the past 3 years.",
            "properties": {
                "2021": {"type": "object"},
                "2022": {"type": "object"},
                "2023": {"type": "object"}
            },
        },
        "text2": {
            "type": "string",
            "description": "Concluding insights and recommendations."
        },
    },
    "required": ["text1", "table", "text2"]
}

# Wrap the LLM with structured output
structured_llm = chat_groq.with_structured_output(json_schema)

app = Flask(__name__)

def get_groq_response(system_message_content, user_input):
    llm = ChatGroq(temperature=0.6, model_name='mixtral-8x7b-32768')
    message = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=user_input)
    ]
    response = structured_llm.invoke(message)
    return response.content

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_dataframe_from_llm(upgraded_prompt):
    system_message = """
    You are a virtual financial expert specializing in CFO-level tasks. Your role is to assist with all financial planning, strategy, reporting, cash flow management, risk mitigation, 
investor relations, fundraising, capital management, and compliance. You will also handle mergers and acquisitions evaluations and lead team management decisions.
Only provide the table containing the relevant data from {response_text}. No additional information is needed regarding the table.
    """
    llm = ChatGroq(temperature=0.6, model_name='mixtral-8x7b-32768')
    message = [
        SystemMessage(content=system_message),
        HumanMessage(content=upgraded_prompt)
    ]
    response = llm.invoke(message)
    try:
        df = pd.read_csv(StringIO(response.content))
        return df
    except pd.errors.ParserError:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_files', methods=['POST'])
def process_files():
    uploaded_files = request.files.getlist('files')
    financial_data = get_pdf_text(uploaded_files)
    company_domain = request.form['company_domain']
    product_focus = request.form['product_focus']
    employee_strength = request.form['employee_strength']
    return render_template('index.html', financial_data=financial_data, company_domain=company_domain,
                           product_focus=product_focus, employee_strength=employee_strength)

@app.route('/submit', methods=['POST'])
def submit_query():
    query = request.form['query']
    company_domain = request.form['company_domain']
    product_focus = request.form['product_focus']
    employee_strength = request.form['employee_strength']
    financial_data = request.form['financial_data']

    prompt_upgrade_system_message_content = f"""
    You are an AI buddy for a CFO, helping them with financial analysis, budgeting, strategy, and business growth.
    Use these details to generate an optimal financial terminology response.
    company domain: {company_domain}, primary product: {product_focus}, employee size: {employee_strength}, User's question: "{query}"
    """

    prompt_to_pass_to_llm = get_groq_response(prompt_upgrade_system_message_content, query)
    ans_human_content = f'''
    {prompt_to_pass_to_llm}. The company's domain is {company_domain}, the current focused product/service is {product_focus}
    and the company size is {employee_strength} and the following is the company's past 3-year financial report: {financial_data}.
    '''
    response_json = get_groq_response(ans_system_message_content, ans_human_content)

    if response_json:
        table_data = response_json.get("table", {})
        if table_data:
            table_df = pd.DataFrame(table_data)
            table_html = table_df.to_html()
        else:
            table_html = "No comparative table provided."

        return jsonify({
            "text1": response_json.get("text1", "No introduction provided."),
            "table": table_html,
            "text2": response_json.get("text2", "No conclusion provided."),
        })

    return jsonify({"error": "No response from LLM"}), 500

if __name__ == '__main__':
    app.run(debug=True)
