import streamlit as st
import pandas as pd
import json
import requests
import time

# Set page config
st.set_page_config(page_title="GPT Prompt Tester", page_icon="🤖", layout="wide")

st.title("🤖 GPT Prompt Testing Tool")
st.markdown(
    "Test multiple GPT prompts against the same email content and compare results"
)

# st.success("✅ Using OpenAI API via HTTP requests (No dependencies required)")

# Initialize session state
if "prompts" not in st.session_state:
    st.session_state.prompts = [""]
if "results" not in st.session_state:
    st.session_state.results = []

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# OpenAI API Key
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Enter your OpenAI API key"
)

# Model selection - Updated with current models
model = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0,
)

# Advanced parameters
st.sidebar.subheader("Model Parameters")
temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    2.0,
    0.15,
    0.05,
    help="Controls randomness in responses. Lower values (0.0-0.3) make responses more focused and deterministic. Higher values (0.7-1.0) make responses more creative and varied.",
)
top_p = st.sidebar.slider(
    "Top P",
    0.0,
    1.0,
    0.2,
    0.05,
    help="Controls diversity by limiting token choices. Lower values (0.1-0.3) focus on most likely words. Higher values (0.8-1.0) allow more diverse word choices.",
)
frequency_penalty = st.sidebar.slider(
    "Frequency Penalty",
    0.0,
    2.0,
    1.0,
    0.1,
    help="Reduces repetition of words/phrases. Higher values (0.5-2.0) discourage repeating the same words. 0 means no penalty for repetition.",
)
presence_penalty = st.sidebar.slider(
    "Presence Penalty",
    0.0,
    2.0,
    0.5,
    0.1,
    help="Encourages talking about new topics. Higher values (0.5-2.0) push the model to introduce new subjects rather than staying on the same topic.",
)
max_tokens = st.sidebar.number_input(
    "Max Tokens",
    100,
    4000,
    1000,
    50,
    help="Maximum number of tokens (words/word pieces) in the response. Higher values allow longer responses but cost more.",
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📧 Email Content")

    # Email payload fields
    content_sender = st.text_input(
        "Sender Email",
        placeholder="sender@example.com",
        help="The email address of the sender",
    )

    content_recipients = st.text_input(
        "Recipient Email(s)",
        placeholder="recipient@example.com, recipient2@example.com",
        help="Comma-separated list of recipient emails",
    )

    content_body = st.text_area(
        "Email Content Body",
        height=200,
        placeholder="Enter the email content here...",
        help="The main body content of the email",
    )

with col2:
    st.header("🔧 Prompt Configuration")

    # Number of prompts
    num_prompts = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=10,
        value=len(st.session_state.prompts),
    )

    # Adjust prompts list based on number
    if num_prompts > len(st.session_state.prompts):
        st.session_state.prompts.extend(
            [""] * (num_prompts - len(st.session_state.prompts))
        )
    elif num_prompts < len(st.session_state.prompts):
        st.session_state.prompts = st.session_state.prompts[:num_prompts]

# Prompt input section
st.header("📝 Enter Your Prompts")

# Create tabs for prompts
if num_prompts > 1:
    tabs = st.tabs([f"Prompt {i+1}" for i in range(num_prompts)])

    for i, tab in enumerate(tabs):
        with tab:
            st.session_state.prompts[i] = st.text_area(
                f"System Prompt {i+1}",
                value=st.session_state.prompts[i],
                height=150,
                placeholder="Enter your system prompt here...",
                key=f"prompt_{i}",
            )
else:
    st.session_state.prompts[0] = st.text_area(
        "System Prompt",
        value=st.session_state.prompts[0],
        height=150,
        placeholder="Enter your system prompt here...",
        key="prompt_0",
    )

# Default prompt button
if st.button("Load Example Cybersecurity Prompt"):
    example_prompt = """You are a cybersecurity specialist. You will objectively analyze emails to assess if there is any indication of cyber threats. Be cautious not to assume threat unless strong evidence is present. Use your analysis of the sender, content, links, and attachments to determine legitimacy or risk. Respond strictly in a valid JSON format. If no threats are detected, clearly state that. Do not exaggerate risk or make speculative assumptions."""
    st.session_state.prompts[0] = example_prompt
    st.rerun()


# Function to call OpenAI API using requests
def call_openai_api_with_requests(
    prompt: str,
    content_sender: str,
    content_recipients: str,
    content_body: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
):
    try:
        # OpenAI API endpoint
        url = "https://api.openai.com/v1/chat/completions"

        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        user_content = f"""Analyze the following email thoroughly and provide detailed information in JSON format:
{{
  "main_intent": String,
  "potentially_malicious": Boolean,
  "threat_type": String | "NA",
  "isFinancialContent": Boolean,
  "isHealthRelatedContent": Boolean,
  "suspiciousLabels": Array of strings,
  "maliciousLabels": Array of strings,
  "trust_score": Integer (out of 100, based on how well the sender's email matches the content and intent)
}}

Here is the email information:
Sender's Email: '{content_sender}',
Recipient's Emails: '{content_recipients}',
Email Content: '{content_body}'
"""

        # Request payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }

        # Make the API request
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(response.text)

        # Check if request was successful
        if response.status_code == 200:
            response_data = response.json()
            raw_response = response_data["choices"][0]["message"]["content"]

            # Try to parse JSON
            try:
                # Clean the response - sometimes models wrap JSON in markdown
                clean_response = raw_response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()

                parsed_response = json.loads(clean_response)
                return {
                    "success": True,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                    "error": None,
                    "usage": response_data.get("usage", {}),
                    "status_code": response.status_code,
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "raw_response": raw_response,
                    "parsed_response": None,
                    "error": f"JSON parsing error: {str(e)}",
                    "usage": response_data.get("usage", {}),
                    "status_code": response.status_code,
                }
        else:
            # Handle API errors
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get(
                    "message", "Unknown API error"
                )
            except:
                error_message = f"HTTP {response.status_code}: {response.text[:200]}"

            return {
                "success": False,
                "raw_response": None,
                "parsed_response": None,
                "error": f"API Error ({response.status_code}): {error_message}",
                "usage": None,
                "status_code": response.status_code,
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "raw_response": None,
            "parsed_response": None,
            "error": "Request timeout - API took too long to respond",
            "usage": None,
            "status_code": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "raw_response": None,
            "parsed_response": None,
            "error": f"Network error: {str(e)}",
            "usage": None,
            "status_code": None,
        }
    except Exception as e:
        return {
            "success": False,
            "raw_response": None,
            "parsed_response": None,
            "error": f"Unexpected error: {str(e)}",
            "usage": None,
            "status_code": None,
        }


# Helper function to format labels for display
def format_labels(labels):
    """Format labels for display in the table"""
    if not labels or not isinstance(labels, list):
        return "None"
    if len(labels) == 0:
        return "None"
    # Join labels with commas, but limit the display length
    formatted = ", ".join(labels)
    if len(formatted) > 100:
        return formatted[:97] + "..."
    return formatted


# Analyze button
st.header("🔍 Analysis")

if st.button("🚀 Analyze All Prompts", type="primary", use_container_width=True):
    # Validation
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
    elif not content_sender or not content_recipients or not content_body:
        st.error("Please fill in all email content fields")
    elif not any(prompt.strip() for prompt in st.session_state.prompts):
        st.error("Please enter at least one prompt")
    else:
        # Run analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        st.session_state.results = []
        valid_prompts = [p for p in st.session_state.prompts if p.strip()]
        total_prompts = len(valid_prompts)

        # Process each prompt
        processed = 0
        for i, prompt in enumerate(st.session_state.prompts):
            if prompt.strip():  # Only process non-empty prompts
                processed += 1
                status_text.text(f"Processing Prompt {processed} of {total_prompts}...")
                progress_bar.progress((processed - 1) / total_prompts)

                # Call API
                result = call_openai_api_with_requests(
                    prompt,
                    content_sender,
                    content_recipients,
                    content_body,
                    api_key,
                    model,
                    temperature,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                    max_tokens,
                )

                result["prompt_number"] = i + 1
                result["prompt_text"] = (
                    prompt[:100] + "..." if len(prompt) > 100 else prompt
                )
                st.session_state.results.append(result)

                # Show intermediate result
                if result["success"]:
                    st.success(f"✅ Prompt {i + 1} completed successfully")
                else:
                    st.error(f"❌ Prompt {i + 1} failed: {result['error']}")

                # Small delay to avoid rate limiting
                time.sleep(1)

        progress_bar.progress(1.0)
        status_text.text("Analysis completed!")
        st.balloons()

# Display results
if st.session_state.results:
    st.header("📊 Results")

    # Show token usage summary
    total_tokens = sum(
        [
            r.get("usage", {}).get("total_tokens", 0) if r.get("usage") else 0
            for r in st.session_state.results
        ]
    )
    successful_calls = len([r for r in st.session_state.results if r["success"]])
    failed_calls = len([r for r in st.session_state.results if not r["success"]])

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful Calls", successful_calls)
    with col2:
        st.metric("Failed Calls", failed_calls)
    with col3:
        st.metric(
            "Total Tokens Used", f"{total_tokens:,}" if total_tokens > 0 else "N/A"
        )

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 Summary Table",
            "📄 Detailed Results",
            "🔍 Raw Responses",
            "📈 Usage Stats",
        ]
    )

    with tab1:
        st.subheader("Summary Comparison Table")

        # Create summary dataframe
        summary_data = []
        for result in st.session_state.results:
            if result["success"] and result["parsed_response"]:
                parsed = result["parsed_response"]
                summary_data.append(
                    {
                        "Prompt": f"Prompt {result['prompt_number']}",
                        "Status": "✅ Success",
                        "Main Intent": str(parsed.get("main_intent", "N/A"))[:50]
                        + (
                            "..."
                            if len(str(parsed.get("main_intent", "N/A"))) > 50
                            else ""
                        ),
                        "Potentially Malicious": parsed.get(
                            "potentially_malicious", "N/A"
                        ),
                        "Threat Type": parsed.get("threat_type", "N/A"),
                        "Trust Score": parsed.get("trust_score", "N/A"),
                        "Financial Content": parsed.get("isFinancialContent", "N/A"),
                        "Health Related": parsed.get("isHealthRelatedContent", "N/A"),
                        "Suspicious Labels": format_labels(
                            parsed.get("suspiciousLabels", [])
                        ),
                        "Malicious Labels": format_labels(
                            parsed.get("maliciousLabels", [])
                        ),
                    }
                )
            else:
                summary_data.append(
                    {
                        "Prompt": f"Prompt {result['prompt_number']}",
                        "Status": "❌ Error",
                        "Main Intent": "ERROR",
                        "Potentially Malicious": "ERROR",
                        "Threat Type": "ERROR",
                        "Trust Score": "ERROR",
                        "Financial Content": "ERROR",
                        "Health Related": "ERROR",
                        "Suspicious Labels": "ERROR",
                        "Malicious Labels": "ERROR",
                    }
                )

        # Create detailed dataframe for CSV export with all information
        detailed_export_data = []
        for result in st.session_state.results:
            base_data = {
                "Prompt_Number": result["prompt_number"],
                "Prompt_Text": st.session_state.prompts[result["prompt_number"] - 1],
                "Status": "Success" if result["success"] else "Error",
                "Model_Used": model,
                "Temperature": temperature,
                "Top_P": top_p,
                "Frequency_Penalty": frequency_penalty,
                "Presence_Penalty": presence_penalty,
                "Max_Tokens": max_tokens,
                "Email_Sender": content_sender,
                "Email_Recipients": content_recipients,
                "Email_Body": content_body,
                "Error_Message": result.get("error", "N/A"),
                "HTTP_Status_Code": result.get("status_code", "N/A"),
                "Raw_Response": result.get("raw_response", "N/A"),
            }

            if result["success"] and result["parsed_response"]:
                parsed = result["parsed_response"]
                analysis_data = {
                    "Main_Intent": parsed.get("main_intent", "N/A"),
                    "Potentially_Malicious": parsed.get("potentially_malicious", "N/A"),
                    "Threat_Type": parsed.get("threat_type", "N/A"),
                    "Trust_Score": parsed.get("trust_score", "N/A"),
                    "Financial_Content": parsed.get("isFinancialContent", "N/A"),
                    "Health_Related": parsed.get("isHealthRelatedContent", "N/A"),
                    "Suspicious_Labels": (
                        ", ".join(parsed.get("suspiciousLabels", []))
                        if isinstance(parsed.get("suspiciousLabels"), list)
                        else str(parsed.get("suspiciousLabels", "N/A"))
                    ),
                    "Malicious_Labels": (
                        ", ".join(parsed.get("maliciousLabels", []))
                        if isinstance(parsed.get("maliciousLabels"), list)
                        else str(parsed.get("maliciousLabels", "N/A"))
                    ),
                }
            else:
                analysis_data = {
                    "Main_Intent": "ERROR",
                    "Potentially_Malicious": "ERROR",
                    "Threat_Type": "ERROR",
                    "Trust_Score": "ERROR",
                    "Financial_Content": "ERROR",
                    "Health_Related": "ERROR",
                    "Suspicious_Labels": "ERROR",
                    "Malicious_Labels": "ERROR",
                }

            # Add token usage data
            usage_data = {}
            if result.get("usage"):
                usage_data = {
                    "Prompt_Tokens": result["usage"].get("prompt_tokens", 0),
                    "Completion_Tokens": result["usage"].get("completion_tokens", 0),
                    "Total_Tokens": result["usage"].get("total_tokens", 0),
                }
            else:
                usage_data = {
                    "Prompt_Tokens": "N/A",
                    "Completion_Tokens": "N/A",
                    "Total_Tokens": "N/A",
                }

            # Combine all data
            detailed_export_data.append({**base_data, **analysis_data, **usage_data})

        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)

            # Create detailed export dataframe
            detailed_df = pd.DataFrame(detailed_export_data)

            # Download buttons with different options
            col1, col2 = st.columns(2)

            with col1:
                # Summary CSV
                csv_summary = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv_summary,
                    file_name="gpt_prompt_analysis_summary.csv",
                    mime="text/csv",
                    help="Download a concise summary table (same as displayed above)",
                )

            with col2:
                # Detailed CSV with all information
                csv_detailed = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Detailed CSV",
                    data=csv_detailed,
                    file_name="gpt_prompt_analysis_detailed.csv",
                    mime="text/csv",
                    help="Download complete analysis data including prompts, parameters, email content, raw responses, and token usage",
                )

    with tab2:
        st.subheader("Detailed Results")

        for result in st.session_state.results:
            status_icon = "✅" if result["success"] else "❌"
            with st.expander(
                f"{status_icon} Prompt {result['prompt_number']} - Detailed Results"
            ):
                st.write("**Prompt Text:**")
                st.code(
                    st.session_state.prompts[result["prompt_number"] - 1],
                    language="text",
                )

                if result["success"]:
                    st.write("**Parsed Response:**")
                    if result["parsed_response"]:
                        # Display as formatted JSON
                        st.json(result["parsed_response"])

                        # Create individual metrics
                        if isinstance(result["parsed_response"], dict):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Trust Score",
                                    result["parsed_response"].get("trust_score", "N/A"),
                                )
                            with col2:
                                malicious = result["parsed_response"].get(
                                    "potentially_malicious", "N/A"
                                )
                                st.metric(
                                    "Potentially Malicious",
                                    (
                                        "Yes"
                                        if malicious
                                        else "No" if malicious is False else malicious
                                    ),
                                )
                            with col3:
                                st.metric(
                                    "Threat Type",
                                    result["parsed_response"].get("threat_type", "N/A"),
                                )
                    else:
                        st.error("Failed to parse JSON response")
                        st.write("**Error:**")
                        st.error(result["error"])
                else:
                    st.error(f"**Error:** {result['error']}")
                    if result.get("status_code"):
                        st.info(f"HTTP Status Code: {result['status_code']}")

                # Show token usage for this prompt
                if result.get("usage"):
                    st.write("**Token Usage:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Prompt Tokens", result["usage"].get("prompt_tokens", 0)
                        )
                    with col2:
                        st.metric(
                            "Completion Tokens",
                            result["usage"].get("completion_tokens", 0),
                        )
                    with col3:
                        st.metric(
                            "Total Tokens", result["usage"].get("total_tokens", 0)
                        )

    with tab3:
        st.subheader("Raw API Responses")

        for result in st.session_state.results:
            status_icon = "✅" if result["success"] else "❌"
            with st.expander(
                f"{status_icon} Prompt {result['prompt_number']} - Raw Response"
            ):
                if result["raw_response"]:
                    st.code(result["raw_response"], language="json")
                else:
                    st.error(f"No response received. Error: {result['error']}")
                    if result.get("status_code"):
                        st.info(f"HTTP Status Code: {result['status_code']}")

    with tab4:
        st.subheader("Usage Statistics")

        # Create token usage dataframe
        token_data = []
        for result in st.session_state.results:
            if result.get("usage"):
                token_data.append(
                    {
                        "Prompt": f"Prompt {result['prompt_number']}",
                        "Status": "Success" if result["success"] else "Failed",
                        "Prompt Tokens": result["usage"].get("prompt_tokens", 0),
                        "Completion Tokens": result["usage"].get(
                            "completion_tokens", 0
                        ),
                        "Total Tokens": result["usage"].get("total_tokens", 0),
                    }
                )
            else:
                token_data.append(
                    {
                        "Prompt": f"Prompt {result['prompt_number']}",
                        "Status": "Failed",
                        "Prompt Tokens": 0,
                        "Completion Tokens": 0,
                        "Total Tokens": 0,
                    }
                )

        if token_data:
            token_df = pd.DataFrame(token_data)
            st.dataframe(token_df, use_container_width=True)

            # Show totals for successful calls only
            successful_df = token_df[token_df["Status"] == "Success"]
            if not successful_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Prompt Tokens", successful_df["Prompt Tokens"].sum()
                    )
                with col2:
                    st.metric(
                        "Total Completion Tokens",
                        successful_df["Completion Tokens"].sum(),
                    )
                with col3:
                    st.metric("Total Tokens", successful_df["Total Tokens"].sum())

                # Estimated cost (approximate - based on current pricing)
                total_tokens_used = successful_df["Total Tokens"].sum()
                if total_tokens_used > 0:
                    if model == "gpt-4o-mini":
                        # Input: $0.15/1M tokens, Output: $0.60/1M tokens (approximate average)
                        estimated_cost = (total_tokens_used / 1000000) * 0.375
                    elif model == "gpt-4o":
                        # Input: $2.50/1M tokens, Output: $10/1M tokens (approximate average)
                        estimated_cost = (total_tokens_used / 1000000) * 6.25
                    elif model == "gpt-4-turbo":
                        # Input: $10/1M tokens, Output: $30/1M tokens (approximate average)
                        estimated_cost = (total_tokens_used / 1000000) * 20
                    else:  # gpt-3.5-turbo
                        # Input: $0.50/1M tokens, Output: $1.50/1M tokens (approximate average)
                        estimated_cost = (total_tokens_used / 1000000) * 1

                    st.info(
                        f"💰 Estimated cost: ${estimated_cost:.6f} USD (approximate)"
                    )
            else:
                st.warning("No successful API calls to calculate usage statistics")
        else:
            st.info("No usage data available")

# Clear results button
if st.session_state.results:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Results"):
            st.session_state.results = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.results = []
            st.session_state.prompts = [""]
            st.rerun()
