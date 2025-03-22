import pandas as pd
import numpy as np
from typing import Dict, List
import json
import os
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

class HotelAnalytics:
    def __init__(self, data_path: str = 'data/hotel_bookings.csv'):
        """Initialize the HotelAnalytics class with data loading and model setup."""
        self.df = None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.data_loaded = False
        
        # Load data and model
        self._load_data(data_path)
        self._load_model()
        
        # Create a summary of the data
        self.data_summary = self._create_data_summary()
    
    def _load_data(self, data_path: str):
        """Load and preprocess the hotel bookings dataset."""
        try:
            # Load data with optimized dtypes and handle NA values
            dtype_dict = {
                'hotel': 'category',
                'arrival_date_year': 'Int16',  # Changed to nullable integer
                'arrival_date_month': 'category',
                'arrival_date_week_number': 'Int8',  # Changed to nullable integer
                'arrival_date_day_of_month': 'Int8',  # Changed to nullable integer
                'stays_in_weekend_nights': 'Int8',  # Changed to nullable integer
                'stays_in_week_nights': 'Int8',  # Changed to nullable integer
                'adults': 'Int8',  # Changed to nullable integer
                'children': 'Int8',  # Changed to nullable integer
                'is_repeated_guest': 'Int8',  # Changed to nullable integer
                'previous_cancellations': 'Int8',  # Changed to nullable integer
                'previous_bookings_not_canceled': 'Int8',  # Changed to nullable integer
                'required_car_parking_spaces': 'Int8',  # Changed to nullable integer
                'total_of_special_requests': 'Int8',  # Changed to nullable integer
                'reservation_status': 'category',
                'reservation_status_date': 'str',
                'customer_type': 'category',
                'adr': 'float32',
                'lead_time': 'Int16',  # Changed to nullable integer
                'market_segment': 'category',
                'distribution_channel': 'category',
                'is_canceled': 'Int8'  # Changed to nullable integer
            }
            
            # Load data with NA handling
            self.df = pd.read_csv(
                data_path,
                dtype=dtype_dict,
                na_values=['NA', 'N/A', ''],
                keep_default_na=True
            )
            
            # Fill NA values with appropriate defaults
            self.df['children'] = self.df['children'].fillna(0)
            self.df['adults'] = self.df['adults'].fillna(1)  # Default to 1 adult
            self.df['required_car_parking_spaces'] = self.df['required_car_parking_spaces'].fillna(0)
            self.df['previous_cancellations'] = self.df['previous_cancellations'].fillna(0)
            self.df['previous_bookings_not_canceled'] = self.df['previous_bookings_not_canceled'].fillna(0)
            self.df['is_repeated_guest'] = self.df['is_repeated_guest'].fillna(0)
            
            # Convert date columns
            self.df['reservation_status_date'] = pd.to_datetime(self.df['reservation_status_date'])
            
            # Add price and financial analysis columns
            self.df['booking_cost'] = 2500  # Base booking cost
            self.df['cancellation_refund'] = self.df['booking_cost'] * 0.30  # 30% refund on cancellation
            
            # Calculate actual revenue per booking
            self.df['actual_revenue'] = np.where(
                self.df['is_canceled'] == 1,
                self.df['booking_cost'] - self.df['cancellation_refund'],
                self.df['booking_cost']
            )
            
            # Calculate total stay duration and guests
            self.df['total_stay'] = self.df['stays_in_weekend_nights'].fillna(0) + self.df['stays_in_week_nights'].fillna(0)
            self.df['total_guests'] = self.df['adults'].fillna(1) + self.df['children'].fillna(0)
            
            # Create indexes for frequently queried columns
            self.df.set_index(['hotel', 'arrival_date_year', 'arrival_date_month'], inplace=True)
            
            self.data_loaded = True
            print("Data loaded successfully with financial metrics")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.model_name = "google/flan-t5-small"  # Free to use, smaller model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.model_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _create_data_summary(self) -> Dict:
        """Create a summary of the data for context."""
        
        df_summary = self.df.reset_index()
        
        summary = {
            "total_bookings": len(df_summary),
            "hotel_types": df_summary['hotel'].unique().tolist(),
            "market_segments": df_summary['market_segment'].unique().tolist(),
            "customer_types": df_summary['customer_type'].unique().tolist(),
            "distribution_channels": df_summary['distribution_channel'].unique().tolist(),
            "average_lead_time": df_summary['lead_time'].mean(),
            "average_adr": df_summary['adr'].mean(),
            "cancellation_rate": (df_summary['is_canceled'].mean() * 100),
            "average_special_requests": df_summary['total_of_special_requests'].mean()
        }
        return summary
    
    def _get_relevant_stats(self, question: str) -> Dict:
        """Get relevant statistics based on the question."""
        stats = {}
        
       
        if "average" in question.lower() and "lead time" in question.lower():
            stats["average_lead_time"] = self.df['lead_time'].mean()
            stats["lead_time_std"] = self.df['lead_time'].std()
            
        elif "cancel" in question.lower():
            stats["cancellation_rate"] = (self.df['is_canceled'].mean() * 100)
            stats["cancellations_by_hotel"] = self.df.groupby('hotel')['is_canceled'].mean().to_dict()
            
        elif "market segment" in question.lower():
            stats["market_segment_distribution"] = self.df['market_segment'].value_counts().to_dict()
            
        elif "special request" in question.lower():
            stats["average_special_requests"] = self.df['total_of_special_requests'].mean()
            stats["special_requests_distribution"] = self.df['total_of_special_requests'].value_counts().to_dict()
            
        elif "distribution channel" in question.lower():
            stats["channel_distribution"] = self.df['distribution_channel'].value_counts().to_dict()
            
        elif "adr" in question.lower() or "rate" in question.lower():
            stats["average_adr"] = self.df['adr'].mean()
            stats["adr_by_hotel"] = self.df.groupby('hotel')['adr'].mean().to_dict()
            
        return stats

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _get_model_response(self, context: str) -> str:
        """Get response from the model with retry logic."""
        try:
            
            inputs = self.tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"\nError generating response: {str(e)}")
            raise e
    
    def answer_question(self, question: str) -> str:
        """Answer a question about the hotel data using LLM."""
        
        stats = self._get_relevant_stats(question)
        
        
        context = f"""
        Here is a summary of the hotel booking data:
        {json.dumps(self.data_summary, indent=2)}
        
        Here are the relevant statistics for your question:
        {json.dumps(stats, indent=2)}
        
        Please answer this question: {question}
        
        Provide a clear and concise answer based on the data provided.
        """
        
        return self._get_model_response(context)

    def get_financial_analysis(self) -> Dict:
        """Get financial analysis of bookings."""
        try:
            
            df_analysis = self.df.reset_index()
            
            analysis = {
                'total_revenue': float(df_analysis['actual_revenue'].sum()),
                'average_revenue_per_booking': float(df_analysis['actual_revenue'].mean()),
                'total_refunds_paid': float((df_analysis['cancellation_refund'] * (df_analysis['is_canceled'] == 1)).sum()),
                'revenue_by_hotel_type': df_analysis.groupby('hotel')['actual_revenue'].agg(['sum', 'mean']).to_dict('index'),
                'monthly_revenue': df_analysis.groupby('arrival_date_month')['actual_revenue'].sum().to_dict(),
                'cancellation_impact': {
                    'total_potential_revenue': float(df_analysis['booking_cost'].sum()),
                    'revenue_lost_to_cancellations': float((df_analysis['cancellation_refund'] * (df_analysis['is_canceled'] == 1)).sum()),
                    'cancellation_rate': float(df_analysis['is_canceled'].mean() * 100)
                },
                'revenue_by_market_segment': df_analysis.groupby('market_segment')['actual_revenue'].agg(['sum', 'mean']).to_dict('index')
            }
            return analysis
        except Exception as e:
            print(f"Error in financial analysis: {str(e)}")
            raise

def main():
    
    print("Initializing the system...")
    analytics = HotelAnalytics()
    print("System initialized successfully!")
    
    
    print("\n" + "="*50)
    print("Welcome to the Hotel Data Analysis Assistant!")
    print("="*50)
    print("\nYou can ask questions about the hotel booking data.")
    print("Example questions you can ask:")
    print("1. What is the average lead time for bookings?")
    print("2. Which hotel type has more cancellations?")
    print("3. What is the most common market segment for bookings?")
    print("4. What is the average number of special requests per booking?")
    print("5. Which distribution channel is most popular for bookings?")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")
    
   
    while True:
        
        question = input("\nYour question: ").strip()
        
        
        if question.lower() in ['quit', 'exit']:
            print("\nThank you for using the Hotel Data Analysis Assistant!")
            break
        
        
        if not question:
            continue
        
        try:
            
            print("\n" + "-"*50)
            print("Answer:")
            answer = analytics.answer_question(question)
            print(answer)
            print("-"*50)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question again.")

if __name__ == "__main__":
    main() 