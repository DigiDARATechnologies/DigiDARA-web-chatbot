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

async function sendMessage(prompt = null, retries = 3) {
    const message = prompt !== null ? prompt : chatInput.value.trim();
    if (message || !chatBody.innerHTML) {
        if (message) {
            const userMessage = document.createElement('div');
            userMessage.className = 'chat-message user-message';
            userMessage.textContent = message;
            chatBody.appendChild(userMessage);
        }

        for (let i = 0; i < retries; i++) {
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include', // Ensure cookies are sent with the request
                    body: JSON.stringify({ prompt: message })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

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
                return; // Success, exit the loop
            } catch (error) {
                console.error(`Attempt ${i + 1} failed:`, error);
                if (i === retries - 1) {
                    // Last retry failed, show error
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'chat-message bot-message';
                    errorMessage.textContent = 'Error connecting to chatbot. Please try again.';
                    chatBody.appendChild(errorMessage);
                }
            }
        }
    }
}

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});