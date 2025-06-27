# DigiDARA-web-chatbot

DigiDARA-web-chatbot is a web-based chatbot application. This project provides a framework for interacting with users through a conversational interface, leveraging various Python scripts and resources for processing data and managing chatbot responses.

## Project Structure

The repository contains the following key files and directories (see screenshot ![image1](image1)):
- **app.py**: Main application entry point.
- **llm_service.py**: Logic for handling interactions with the language model.
- **lead_processing.py**: Handles processing of lead data.
- **upload_to_qdrant.py**: Script for uploading data to Qdrant vector database.
- **last_processed.json** & **processed_emails.json**: JSON files for storing processed information.
- **requirements.txt**: Lists the Python dependencies for the project.
- **static/**: Directory for static assets (CSS, JS, images, etc.).
- **templates/**: Directory for HTML templates.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ranjeeth11/DigiDARA-web-chatbot.git](https://github.com/DigiDARATechnologies/DigiDARA-web-chatbot)
   cd DigiDARA-web-chatbot
   ```

2. **Create a virtual environment (optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the chatbot application, run:
```bash
python app.py
```
The application will start a local web server. Open your browser and navigate to the displayed URL (e.g., http://localhost:5000).

## Features

- Conversational chatbot interface
- Can process leads and store processed data
- Integration with Qdrant vector database
- Modular Python codebase

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

Specify your license here (e.g., MIT, Apache 2.0, etc.).
