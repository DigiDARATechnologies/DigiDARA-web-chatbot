// Chat widget functionality
const chatContainer = document.getElementById('chatContainer');
const chatBody = document.getElementById('chatBody');
const chatInput = document.getElementById('chatInput');

function toggleChat() {
    chatContainer.style.display = chatContainer.style.display === 'none' ? 'flex' : 'none';
    if (chatContainer.style.display === 'flex' && !chatBody.innerHTML) {
        sendMessage(''); // Trigger welcome message on first open
    }
}

async function sendMessage(prompt = null) {
    const message = prompt !== null ? prompt : chatInput.value.trim();
    if (message || !chatBody.innerHTML) {
        if (message) {
            const userMessage = document.createElement('div');
            userMessage.className = 'chat-message user-message';
            userMessage.textContent = message;
            chatBody.appendChild(userMessage);
        }

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: message })
            });
            const data = await response.json();

            const botMessage = document.createElement('div');
            botMessage.className = 'chat-message bot-message';
            botMessage.textContent = data.response;
            chatBody.appendChild(botMessage);

            if (data.quickReplies) {
                const quickRepliesDiv = document.createElement('div');
                quickRepliesDiv.className = 'quick-replies';
                data.quickReplies.forEach(reply => {
                    const button = document.createElement('button');
                    button.className = 'quick-reply-btn';
                    button.textContent = reply;
                    button.onclick = () => sendMessage(reply);
                    quickRepliesDiv.appendChild(button);
                });
                chatBody.appendChild(quickRepliesDiv);
            }

            chatBody.scrollTop = chatBody.scrollHeight;
            chatInput.value = '';
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = document.createElement('div');
            errorMessage.className = 'chat-message bot-message';
            errorMessage.textContent = 'Error connecting to chatbot.';
            chatBody.appendChild(errorMessage);
        }
    }
}

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});