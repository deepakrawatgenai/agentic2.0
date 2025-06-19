import streamlit as st
from dotenv import load_dotenv
import os

from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.google_places_api import GooglePlacesAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# env_loader.py
def load_env_variables():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# tools_module.py

from langchain_core.tools import tool
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.google_places_api import GooglePlacesAPIWrapper

# ---------------------------- Weather Tools ----------------------------

class WeatherTools:
    """
    Provides weather-related utilities including current weather and forecast.
    """

    @tool
    def get_current_weather(self, city: str) -> str:
        """
        Returns the current weather conditions for a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: Weather report as a string.
        """
        weather = OpenWeatherMapAPIWrapper()
        return weather.run(city)

    @tool
    def get_weather_forecast(self, city: str) -> str:
        """
        Returns a short-term weather forecast for a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: Weather forecast details.
        """
        return OpenWeatherMapAPIWrapper().run(city)

# ---------------------------- Location & POI Tools ----------------------------

class LocationTools:
    """
    Provides tools to discover local attractions, restaurants, and activities.
    """

    @tool
    def search_attraction(self, city: str) -> str:
        """
        Searches for top tourist attractions in a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: List of top attractions.
        """
        places = GooglePlacesAPIWrapper()
        return places.run(city + " top tourist attractions")

    @tool
    def search_restaurant(self, city: str) -> str:
        """
        Searches for best-rated restaurants in a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: List of restaurants.
        """
        places = GooglePlacesAPIWrapper()
        return places.run(city + " best restaurants")

    @tool
    def search_activity(self, city: str) -> str:
        """
        Searches for popular activities to do in a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: List of activities.
        """
        return TavilySearchResults().run(f"Top fun activities in {city}")

# ---------------------------- TransportTools Tools ----------------------------

class TransportTools:
    """
    Provides Transport avalblity in the city functionalities.
    """

    @tool
    def search_transport(self, city: str) -> str:
        """
        Retrieves Transport option avalible to reach the city.

        Args:
            city (str): Name of the city.

        Returns:
            str: details of transport available in city .
        """
        return TavilySearchResults().run(f"Transport availble in the city {city}")
    
# ---------------------------- Accommodation Tools ----------------------------

class HotelTools:
    """
    Provides hotel search and price estimation functionalities.
    """

    @tool
    def search_hotel(self, city: str) -> str:
        """
        Retrieves average hotel costs per night in a given city.

        Args:
            city (str): Name of the city.

        Returns:
            str: Average hotel cost per night.
        """
        return TavilySearchResults().run(f"Average hotel cost per night in {city}")

# ---------------------------- Currency & Budget Tools ----------------------------

class CurrencyTools:
    """
    Provides currency exchange rate and conversion utilities.
    """

    @tool
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> str:
        """
        Retrieves the current exchange rate between two currencies.

        Args:
            from_currency (str): Base currency code.
            to_currency (str): Target currency code.

        Returns:
            str: Current exchange rate as a string.
        """
        return TavilySearchResults().run(f"Exchange rate {from_currency} to {to_currency}")

    @tool
    def convert_currency(self, amount: float, rate: float) -> float:
        """
        Converts a currency amount using the given exchange rate.

        Args:
            amount (float): Amount in base currency.
            rate (float): Exchange rate.

        Returns:
            float: Converted amount.
        """
        return amount * rate

# ---------------------------- Itinerary Tools ----------------------------

class ItineraryTools:
    """
    Provides daily planning and trip summary generation.
    """

    @tool
    def create_day_plan(self, city: str) -> str:
        """
        Generates a daily itinerary for the specified city.

        Args:
            city (str): Name of the city.

        Returns:
            str: Day-wise plan.
        """
        return TavilySearchResults().run(f"Day-wise trip itinerary for {city}")

    @tool
    def create_summary(self, city: str, weather: str, itinerary: str, cost: float, currency: str) -> str:
        """
        Generates a human-readable trip summary.

        Args:
            city (str): Travel city.
            weather (str): Weather info.
            itinerary (str): Itinerary text.
            cost (float): Total cost.
            currency (str): Currency code.

        Returns:
            str: Summary text.
        """
        return f"Trip to {city}: Weather - {weather}, Plan - {itinerary}, Cost - {cost} {currency}"

# llm_module.py

def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# agent_module.py

class TravelAgentAssistant:
    def __init__(self, llm, tools):
        self.system_prompt = '''You are a highly skilled and helpful travel assistant agent. Your job is to provide complete and accurate travel information using a set of tools available to you.

Use the tools as needed based on the user's query. You can:
- Search real-time weather and forecast for any location
- Find hotels and estimate their costs
- Suggest attractions, restaurants, and activities
- Convert currencies and estimate budgets
- Plan itineraries day-by-day
- Provide the total trip cost and create a summary
- Search the internet (via Tavily) for travel alerts or up-to-date information

Only use tools when necessary, and combine multiple tools to answer complex questions. Respond clearly and in a friendly, informative tone. When returning travel plans or cost estimations, be concise, structured, and user-friendly.

You have access to the following tools:
    get_current_weather,
    get_weather_forecast,
    search_attraction,
    search_restaurant,
    search_activity,
    search_hotel,
    get_exchange_rate,
    convert_currency,
    create_day_plan,
    create_summary

Decide which tools to use based on the question. If the question can be answered without using a tool, do so directly. Otherwise, call the appropriate tools with correct parameters.

Your goal is to make trip planning seamless and delightful.
'''
        self.llm = llm
        self.tools = tools
        self.llm_with_tool = self.llm.bind_tools(self.tools)
        self.graph = StateGraph(MessagesState)
        self._setup_graph()

    def function_1(self, state: MessagesState):
        user_question = state["messages"]
        input_question = [self.system_prompt] + user_question
        response = self.llm_with_tool.invoke(input_question)
        return {"messages": [response]}

    def _setup_graph(self):
        self.graph.add_node("llm_decision_step", self.function_1)
        self.graph.add_node("tools", ToolNode(self.tools))
        self.graph.add_edge(START, "llm_decision_step")
        self.graph.add_conditional_edges("llm_decision_step", tools_condition)
        self.graph.add_edge("tools", "llm_decision_step")
        self.graph.add_edge("tools", "llm_decision_step")
        self.react1 = self.graph.compile()

    def plan_trip(self, user_input):
        result = self.react1.invoke({"input": user_input})
        return result

# main.py (example usage in your Streamlit app)

load_env_variables()

tools = [
    WeatherTools.get_current_weather,
    WeatherTools.get_weather_forecast,
    LocationTools.search_attraction,
    LocationTools.search_restaurant,
    LocationTools.search_activity,
    TransportTools.search_transport,
    HotelTools.search_hotel,
    CurrencyTools.get_exchange_rate,
    CurrencyTools.convert_currency,
    ItineraryTools.create_day_plan,
    ItineraryTools.create_summary
]
llm = get_llm()
assistant = TravelAgentAssistant(llm, tools)




st.set_page_config(page_title="AI Travel Planner", page_icon="🌍")
st.title("🧳 AI Travel Assistant & Expense Planner")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Where are you planning your next trip?")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Planning your trip..."):
        result = assistant.plan_trip({"input": user_input})
        st.session_state.chat_history.append(("bot", result["output"] if "output" in result else "Sorry, I couldn't fetch that."))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)