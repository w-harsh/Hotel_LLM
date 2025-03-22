from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from hotel_analytics import HotelAnalytics
import traceback

app = Flask(__name__)
CORS(app)  


print("Initializing the Hotel Analytics system...")
analytics = HotelAnalytics()  
print("System initialized successfully!")

@app.route('/', methods=['GET'])
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/analytics', methods=['POST'])
def get_analytics():
    """
    Generate analytics report based on specified metrics.
    
    Expected JSON payload:
    {
        "metrics": ["cancellation_rate", "average_lead_time", "market_segments", "hotel_types", "adr_by_hotel", "special_requests"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'metrics' not in data:
            return jsonify({
                'error': 'Invalid request. Please provide metrics in the request body.',
                'example_request': {
                    'metrics': ['cancellation_rate', 'average_lead_time']
                }
            }), 400
            
        metrics = data['metrics']
        if not isinstance(metrics, list):
            return jsonify({
                'error': 'Metrics should be a list of strings.',
                'example_request': {
                    'metrics': ['cancellation_rate', 'average_lead_time']
                }
            }), 400
            
       
        results = {}
        for metric in metrics:
            if metric == 'cancellation_rate':
                results[metric] = analytics.df['is_canceled'].mean() * 100
            elif metric == 'average_lead_time':
                results[metric] = analytics.df['lead_time'].mean()
            elif metric == 'market_segments':
                results[metric] = analytics.df['market_segment'].value_counts().to_dict()
            elif metric == 'hotel_types':
                results[metric] = analytics.df['hotel'].value_counts().to_dict()
            elif metric == 'adr_by_hotel':
                results[metric] = analytics.df.groupby('hotel')['adr'].mean().to_dict()
            elif metric == 'special_requests':
                results[metric] = {
                    'average': analytics.df['total_of_special_requests'].mean(),
                    'distribution': analytics.df['total_of_special_requests'].value_counts().to_dict()
                }
            else:
                results[metric] = f"Metric '{metric}' not available"
                
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Answer questions about the hotel bookings data.
    
    Expected JSON payload:
    {
        "question": "What is the average lead time for bookings?"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Invalid request. Please provide a question in the request body.',
                'example_request': {
                    'question': 'What is the average lead time for bookings?'
                }
            }), 400
            
        question = data['question']
        if not isinstance(question, str):
            return jsonify({
                'error': 'Question should be a string.',
                'example_request': {
                    'question': 'What is the average lead time for bookings?'
                }
            }), 400
            
        # Get answer from the model
        answer = analytics.answer_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/financial-analysis', methods=['GET'])
def get_financial_analysis():
    """Get financial analysis of hotel bookings."""
    try:
        analysis = analytics.get_financial_analysis()
        return jsonify({
            'success': True,
            'data': {
                'total_revenue': f"₹{analysis['total_revenue']:,.2f}",
                'average_revenue_per_booking': f"₹{analysis['average_revenue_per_booking']:,.2f}",
                'total_refunds_paid': f"₹{analysis['total_refunds_paid']:,.2f}",
                'revenue_by_hotel_type': {
                    hotel: {
                        'total_revenue': f"₹{data['sum']:,.2f}",
                        'average_revenue': f"₹{data['mean']:,.2f}"
                    }
                    for hotel, data in analysis['revenue_by_hotel_type'].items()
                },
                'monthly_revenue': {
                    month: f"₹{revenue:,.2f}"
                    for month, revenue in analysis['monthly_revenue'].items()
                },
                'cancellation_impact': {
                    'total_potential_revenue': f"₹{analysis['cancellation_impact']['total_potential_revenue']:,.2f}",
                    'revenue_lost_to_cancellations': f"₹{analysis['cancellation_impact']['revenue_lost_to_cancellations']:,.2f}",
                    'cancellation_rate': f"{analysis['cancellation_impact']['cancellation_rate']:.1f}%"
                },
                'revenue_by_market_segment': {
                    segment: {
                        'total_revenue': f"₹{data['sum']:,.2f}",
                        'average_revenue': f"₹{data['mean']:,.2f}"
                    }
                    for segment, data in analysis['revenue_by_market_segment'].items()
                }
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': analytics.model_name,
        'uptime': 'OK',
        'database': 'OK'
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation at the root endpoint (/)'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The method {request.method} is not allowed for this endpoint'
    }), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 