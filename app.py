import os
import streamlit as st
import pandas as pd
from io import StringIO
from langchain.schema import SystemMessage, HumanMessage
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from pandasai import Agent
from pandasai.llm import OpenAI, BambooLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
# Load environment variables
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize ChatGroq (or another compatible LLM)
chat_groq = ChatGroq(model='mixtral-8x7b-32768')
llm = BambooLLM(api_key='$2a$10$nK8sKNE9iEnUmTlhKaX17eDjddKkpYSK0yEi2Ph5n77FQN0d/PyMy')


def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs."""
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text is extracted for the page
                text += page_text
            else:
                st.warning("A page in the PDF couldn't be read.")
    return text

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

def chunk_text(text, chunk_size=4000, chunk_overlap=0):
    """Split text into chunks that fit within token limits."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def summarize_financial_data(text):
    """Create a condensed version of the financial data focusing on key metrics."""
    if len(text) > 4000:  # If text is too long
        chunks = chunk_text(text)
        # Take first chunk and add a note
        return chunks[0][:4000] + "\n\n[Note: This is a condensed version of the financial data focusing on key information.]"
    return text

def get_groq_response(system_message_content, user_input):
    """Get structured financial analysis response from LLM with chunking."""
    # If the user input is too large, chunk it
    if len(user_input) > 400000:  # Conservative limit to account for system message
        chunks = chunk_text(user_input)
        # Process first chunk only or implement logic to combine multiple chunks
        user_input = chunks[0] + "\n\n[Note: Due to length limitations, this is a partial analysis of the document.]"
    
    message = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=user_input)
    ]
    
    try:
        response = structured_llm.invoke(message)
        return response
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None

# Streamlit Interface
    st.set_page_config(layout="wide", page_title='CFO Buddy')
st.title('CFO Buddy: Your Financial Analysis Assistant')
st.warning("This is a Beta Version Application. Please use Annual Financial Report Only")

# Sidebar: Upload Financial Reports and Additional Information
financial_data = ''  # Initialize as empty string
with st.sidebar:
    st.sidebar.header('Upload Financial Reports and Company Info')
    uploaded_files = st.sidebar.file_uploader('Upload past 3 years Financial Reports', type='pdf', accept_multiple_files=True)
    if st.sidebar.button("Process Files"):
        with st.spinner('Processing...'):
            if 'financial_data' not in st.session_state:
                st.session_state.financial_data = ''
            st.session_state.financial_data = get_pdf_text(uploaded_files)
            if st.session_state.financial_data:
                st.sidebar.success('PDFs uploaded and processed successfully.')
            else:
                st.sidebar.error('Failed to read text from the PDFs.')



    company_domain = st.sidebar.text_input('Company Domain')
    product_focus = st.sidebar.text_input('Primary Product')
    employee_strength = st.sidebar.selectbox('Employee Strength', ('0-10', '10-100', '100-1,000', '1,000-10,000', '10,000 +'))

print(financial_data)
# Taking user input
query = st.text_area('Ask Your financial question here...')

prompt_upgrade_system_message_content = f"""
You are an AI buddy for a CFO, helping them with financial analysis, budgeting,
    strategy, and business growth. Follow a structured thought process for every response.

    Here, your sole responsibility is to upgrade the {query} in the optimal financial terminology 
    in the following structured thought process.
    Use these information for generating better output 
    company domain : {company_domain},
    primary product: {product_focus},
    employee size : {employee_strength}
    User's question: "{query}"

    Chain of thought:

    1. **Identify the Core Financial Question**:
        - Recognize the main issue, question, or objective in the user's query.
        - For example, is the question about budgeting, cost management, revenue forecasting, etc.?

    2. **Contextual Understanding**:
        - Consider the broader financial context: company goals, market conditions, financial policies,
          or specific department metrics related to the question.
        - For instance, if the query is about cost optimization, assess which costs are typically high in the industry.

    3. **Analyze Relevant Financial Data**:
        - Break down any financial calculations, estimates, or assumptions needed.
        - Highlight any relevant financial metrics, ratios, or KPIs that will aid in answering the question.
        - If specific data isn't provided, mention assumptions and common industry benchmarks.

    4. **Generate Insightful Recommendations**:
        - Offer clear, actionable advice based on the analysis.
        - Specify if certain actions are likely to improve revenue, reduce costs, or align with strategic goals.
        - Suggest further areas of analysis if more data or investigation is required.

    5. **Summarize and Conclude**:
        - Recap the primary insights and recommendations in a concise manner.
        - Encourage follow-up questions to clarify or expand on certain points.
Example:
{{
    "text1": "Introduction and insights...",
    "table": {{ "2021": {{}}, "2022": {{}}, "2023": {{}} }},
    "text2": "Summary and recommendations."
}}

Company domain: {company_domain}, Product: {product_focus}, Employee size: {employee_strength}, Query: "{query}"
"""

ans_system_message_content = """
You are a virtual financial expert specializing in CFO-level tasks. Your role is to assist with all financial planning, strategy, reporting, cash flow management, risk mitigation, 
investor relations, fundraising, capital management, and compliance. You will also handle mergers and acquisitions evaluations and lead team management decisions.

Answer every question with detailed explanations, calculations where necessary, and clear insights to support business decision-making.
When responding:

1. Respond with step-by-step calculation of the figures
2. Provide actionable recommendations when addressing strategic and financial decisions.
3. Use simple, plain language when explaining complex financial concepts to non-experts.
4. Always include context-based reasoning for any decisions or recommendations.
5. For forecasting or planning tasks, consider current market trends and company data.
6. Ensure clarity in reports and documentation for investor relations and internal communication.
7. Make sure to do a competitive analysis of the relevant figures with the previous years.

IMPORTANT: Your response must be a valid JSON object with exactly this structure:
{
    "text1": "Your initial analysis and insights here as a string",
    "table": {
        "2021": {"metric1": "value1", "metric2": "value2"},
        "2022": {"metric1": "value1", "metric2": "value2"},
        "2023": {"metric1": "value1", "metric2": "value2"}
    },
    "text2": "Your conclusions and recommendations here as a string"
}
"""

def analyze_financial_data(text, max_chunks=3):
    """Analyze multiple chunks of financial data with summarization."""
    chunks = chunk_text(text, chunk_size=4000)
    all_analyses = []
    
    # First summarize each chunk
    for i, chunk in enumerate(chunks[:max_chunks]):
        # Get a summary of the chunk first
        summary_prompt = f"Summarize the following financial data, focusing on key metrics and insights: {chunk}"
        summary = get_groq_response(
            "You are a financial summarizer. Create a concise summary focusing on key financial metrics and insights.",
            summary_prompt
        )
        
        if summary:
            # Now analyze the summarized content
            ans_human_content = f'''Analysis for summarized section {i+1}: 
            Company in {company_domain} sector, 
            focusing on {product_focus}, size: {employee_strength}. 
            Financial data section: {summary.get("text1", "")}'''
            
            response = get_groq_response(ans_system_message_content, ans_human_content)
            if response:
                all_analyses.append(response)
    
    return combine_analyses(all_analyses)

def combine_analyses(analyses_list):
    """Combine multiple analysis results into a single coherent response."""
    if not analyses_list:
        return None
    
    # Return first analysis if only one exists
    if len(analyses_list) == 1:
        return analyses_list[0]
    
    # Combine multiple analyses
    combined = {
        "text1": "\n".join(a.get("text1", "") for a in analyses_list),
        "table": analyses_list[0].get("table", {}),  # Use first analysis's table
        "text2": "\n".join(a.get("text2", "") for a in analyses_list)
    }
    
    return combined

def extract_key_financials(text):
    """Extract only the most important financial information."""
    key_patterns = [
        r'Revenue:?\s*[\d,\.]+',
        r'Net Income:?\s*[\d,\.]+',
        r'EBITDA:?\s*[\d,\.]+',
        r'Total Assets:?\s*[\d,\.]+',
        r'Net Profit Margin:?\s*[\d,\.]+%?',
        # Add more patterns as needed
    ]
    
    summary = []
    for pattern in key_patterns:
        matches = re.findall(pattern, text)
        if matches:
            summary.extend(matches)
    
    return "\n".join(summary)

def hierarchical_summarize(text):
    """Summarize text in multiple stages for better context retention."""
    chunks = chunk_text(text, chunk_size=4000)
    
    # First level summaries
    section_summaries = []
    for chunk in chunks:
        summary = get_groq_response(
            "Summarize this financial text in 200 words, focusing on key metrics:",
            chunk
        )
        if summary:
            section_summaries.append(summary)
    
    # Final summary
    final_summary = "\n".join(section_summaries)
    if len(final_summary) > 4000:
        final_summary = get_groq_response(
            "Create a final summary of these financial insights in 500 words:",
            final_summary
        )
    
    return final_summary

if st.button('Submit') and query:
    financial_data = st.session_state.get('financial_data', '')
    
    # Choose one of the approaches above:
    # Option 1: Multiple chunk analysis
    analysis = analyze_financial_data(financial_data)
    
    # Option 2: Key financial extraction
    financial_summary = extract_key_financials(financial_data)
    
    # Option 3: Hierarchical summarization
    financial_summary = hierarchical_summarize(financial_data)
    
    # Then proceed with the analysis using the summarized data
    prompt_to_pass_to_llm = get_groq_response(prompt_upgrade_system_message_content, query)
    if prompt_to_pass_to_llm:
        ans_human_content = f'''Analysis based on key financial metrics: 
        Company: {company_domain}
        Product: {product_focus}
        Size: {employee_strength}
        Financial Summary: {financial_summary}'''
        print("============Debugging0=====================",len(financial_data))
        response_json = get_groq_response(ans_system_message_content, ans_human_content)

        if response_json:
            # Display components of the JSON
            st.subheader("Analysis")
            st.write(response_json.get("text1", "No introduction provided."))

            st.subheader("Comparative Table")
            table_data = response_json.get("table", {})
            if table_data:
                table_df = pd.DataFrame(table_data)
                st.write(table_df)  
            else:
                st.write("No comparative table provided.")

            st.subheader("Conclusion")
            st.write(response_json.get("text2", "No conclusion provided."))
            
            # Generating visualizations based on user query with PandasAI
            if not table_df.empty:
                st.subheader("Visualization based on your query")
                visualization_query= 'make a relevant chart out of th   is dataframe'
    
                smart_df= Agent(table_df,config={'llm':llm})
                try:
                    # Use .chat to ask for a visualization
                    chart = smart_df.chat(visualization_query)  # Generate visualization with .chat method
                    chart_path = 'exports/charts/temp_chart.png'  # Default path PandasAI uses
                    if os.path.exists(chart_path):
                        st.image(chart_path, caption="Generated Chart", use_column_width=True) # Display the chart in Streamlit
                except Exception as e:
                    st.error(f"Could not generate visualization: {e}") 
        
        
