# app.py - Main Flask application for Chroma-Mood

from flask import Flask, render_template, request

# Create a Flask web app instance
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both GET and POST requests for the main page.
    - GET: Displays the main page with the form.
    - POST: Processes the text input and displays the emotion and color.
    """
    # Initialize variables to be passed to the template
    user_text = ""
    emotion = None
    color = None

    if request.method == 'POST':
        # Get the text from the form submission
        user_text = request.form.get('user_text', '')

        # TODO: Add emotion detection logic here
        # For now, we'll use a hardcoded dummy result
        if user_text:
            emotion = "Joy"
            color = "#FFD700"  # Gold

    # Render the main page, passing the variables to the template
    return render_template('index.html', user_text=user_text, emotion=emotion, color=color)

if __name__ == '__main__':
    # Run the app in debug mode for development
    app.run(debug=True)
