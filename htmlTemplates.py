css = """
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1rem;
    background-color: #f7f7f8;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
.chat-message.user {
    background-color: #155eef;
    color: #ffffff;
}
.chat-message.bot {
    background-color: #fff4d6;
    color: #202124;
}
.chat-message .avatar {
    width: 64px;
    flex: 0 0 64px;
}
.chat-message .avatar img {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    flex: 1;
    line-height: 1.5;
    white-space: pre-wrap;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""
