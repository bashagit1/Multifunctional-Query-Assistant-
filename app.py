import streamlit as st
import openai
import os
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.tools.json_helpers import extract_json_from_text

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM instance
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")

# Define Agent Class
class Agent:
    def __init__(self, model_provider, model_name):
        self.llm_instance = LLM.create(model_provider, model_name=model_name)
        self.available_actions = {}

        # Template for guiding the agent's actions
        self.system_prompt_template = """
            You run in a loop of Thought, Action, PAUSE, Action_Response.
            At the end of the loop, output an Answer.

            Use Thought to understand the question you have been asked.
            Use Action to run one of the actions available to you - then return PAUSE.
            Action_Response will be the result of running those actions.

            Available actions:
            {actions_list}
        """

    def add_tool(self, tool_function):
        tool_name = tool_function.__name__
        description = tool_function.__doc__.strip()
        self.available_actions[tool_name] = {"function": tool_function, "description": description}

    def construct_system_prompt(self):
        actions_description = "\n".join(
            [f"{name}:\n {details['description']}" for name, details in self.available_actions.items()]
        )
        return self.system_prompt_template.format(actions_list=actions_description)

    def generate_response(self, user_query, max_turns=5):
        react_system_prompt = self.construct_system_prompt()
        messages = [{"role": "system", "content": react_system_prompt},
                    {"role": "user", "content": user_query}]
        turn_count = 1

        while turn_count <= max_turns:
            turn_count += 1
            agent_response = self.llm_instance.generate_response(messages=messages)
            messages.append({"role": "assistant", "content": agent_response})

            # Extract action JSON from text response
            action_json = extract_json_from_text(agent_response)
            if action_json:
                function_name = action_json[0]['function_name']
                function_params = action_json[0]['function_params']
                action_function = self.available_actions[function_name]["function"]
                result = action_function(**function_params)
                function_result_message = f"Action_Response: {result}"
                messages.append({"role": "user", "content": function_result_message})
            else:
                break
        return messages[-1]["content"]

# Define some tools/actions for the agent
def summarize_content(content):
    """Summarize the provided content."""
    return llm_instance.generate_response(f"Summarize this content: {content}")

def generate_blog_ideas(topic, style):
    """Generate blog ideas for a specific topic in a certain style."""
    return llm_instance.generate_response(f"Generate 5 blog ideas about {topic} in a {style} style.")

# Initialize Agent and add tools
agent = Agent(model_provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")
agent.add_tool(summarize_content)
agent.add_tool(generate_blog_ideas)

# Streamlit UI with Enhanced Styling
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🤖💡 Multifunctional Query Assistant 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4B0082;'>Ask me anything! I can summarize, create ideas, analyze, and more. 📝</p>", unsafe_allow_html=True)

# Sidebar - AI Assistant Options with Styled Headers
with st.sidebar:
    st.markdown("<h2 style='color: #FFA07A;'>🛠️ Options Panel</h2>", unsafe_allow_html=True)
    response_style = st.selectbox("🎨 Response Style", ["Concise", "Detailed", "SEO-optimized"])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("👈 Adjust settings here!")

# User Input Section
st.markdown("<h3 style='color: #4B0082;'>📥 Enter Your Query Below:</h3>", unsafe_allow_html=True)
user_query = st.text_area("Type a question or task here:", placeholder="e.g., Provide a detailed analysis on climate change")

# Generate Response
if st.button("✨ Generate Response"):
    if user_query:
        # Generate response using the agent
        response = agent.generate_response(user_query)
        st.markdown("<h3 style='color: #4B0082;'>🤖 AI Agent Response:</h3>", unsafe_allow_html=True)
        st.write(response)
        # Option to download
        st.download_button(label="💾 Download Response", data=response, file_name="ai_response.txt")
    else:
        st.warning("Please enter a query to generate a response!")

# Display User Instructions
st.info("**🛠️ How to Use:**\n1. Enter a topic or question.\n2. Select the response style in the sidebar.\n3. Click 'Generate Response' to get your answer!")
