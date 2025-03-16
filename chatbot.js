async function sendMessage(query, user_id = "anonymous", preferred_language = "en-IN") {
  const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
          "Content-Type": "application/json",
      },
      body: JSON.stringify({
          query: query,
          user_id: user_id,
          preferred_language: preferred_language,
      }),
  });

  if (!response.ok) {
      throw new Error("Failed to fetch response from chatbot");
  }

  const data = await response.json();
  return data.response;
}

// Example usage
document.getElementById("send-button").addEventListener("click", async () => {
  const query = document.getElementById("chat-input").value;
  const response = await sendMessage(query, "user123", "hi-IN");
  document.getElementById("chat-response").innerText = response;
});