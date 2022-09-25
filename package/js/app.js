const output = document.getElementById('output')

async function send_request(text) {
	output.innerText = "Generating..."
	const response = await fetch('/compute', {
		method: "POST",
		body: JSON.stringify({
			text
		}),
		headers: {
			'Content-Type': 'application/json'
		}
	})
	const data = await response.json()
	output.innerText = data[2]
}

const form = document.getElementById('para_form')
form.addEventListener("submit", (event) => {
	event.preventDefault()
	send_request(form.elements['text'].value)
});
