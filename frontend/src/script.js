function clearInput() {
    const inputs = document.querySelectorAll("#inputs-container input");
    inputs.forEach(input => input.value = "");

    document.getElementById("mood-section").classList.add("hidden");
    document.getElementById("recommended-section").classList.add("hidden");

    const inputContainer = document.getElementById("inputs-container");
    inputContainer.classList.remove("col-span-2");
    inputContainer.classList.add("col-span-3");

    chart.updateOptions({
        series: [0],
        labels: ["🎵"]
    });

    const recContainer = document.getElementById("recommended-songs-container");
    if (recContainer) recContainer.innerHTML = "";

    const moodElem = document.getElementById("mood-text");
    if (moodElem) moodElem.innerText = "Mood";

    const percentageElem = document.querySelector("#mood-section h1.text-3xl");
    if (percentageElem) percentageElem.innerText = "0%";
}

// Validation function
function validateInputs() {
    const inputData = [
        { id: "danceability", min: 0, max: 1 },
        { id: "energy", min: 0, max: 1 },
        { id: "loudness", min: -60, max: 0 },
        { id: "speechiness", min: 0, max: 10 },
        { id: "acousticness", min: 0, max: 1 },
        { id: "instrumentalness", min: 0, max: 1 },
        { id: "liveness", min: 0, max: 1 },
        { id: "valence", min: 0, max: 1 },
        { id: "tempo", min: 0, max: 100 }
    ];

    for (const input of inputData) {
        const elem = document.getElementById(input.id);
        const value = parseFloat(elem.value);
        if (isNaN(value) || value < input.min || value > input.max) {
            alert(`Invalid value for ${input.id}. Must be between ${input.min} and ${input.max}.`);
            elem.focus();
            return false;
        }
    }
    return true;
}

async function predictMood() {
    if (!validateInputs()) return;

    const features = {
        danceability: parseFloat(document.getElementById("danceability").value),
        energy: parseFloat(document.getElementById("energy").value),
        loudness: parseFloat(document.getElementById("loudness").value),
        speechiness: parseFloat(document.getElementById("speechiness").value),
        acousticness: parseFloat(document.getElementById("acousticness").value),
        instrumentalness: parseFloat(document.getElementById("instrumentalness").value),
        liveness: parseFloat(document.getElementById("liveness").value),
        valence: parseFloat(document.getElementById("valence").value),
        tempo: parseFloat(document.getElementById("tempo").value),
    };

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(features)
        });

        const data = await res.json();
        console.log("Prediction result:", data);

        const percentage = data.confidence;
        const mood = data.mood;
        const recommendedSongs = data.recommended_songs || [];

        const moodEmojiMap = {
            happy: "😊",
            sad: "😢",
            calm: "😌",
            energetic: "🥳",
            neutral: "😐"
        };
        const emoji = moodEmojiMap[mood] || "🎵";

        document.getElementById("mood-section").classList.remove("hidden");
        document.getElementById("recommended-section").classList.remove("hidden");

        const inputContainer = document.getElementById("inputs-container");
        inputContainer.classList.remove("col-span-3");
        inputContainer.classList.add("col-span-2");

        chart.updateOptions({
            series: [percentage],
            labels: [emoji],
        });

        const moodContainer = document.querySelector("#mood-section .flex.flex-col.items-center.relative");
        if (moodContainer) {
            const percentageElem = moodContainer.querySelector("h1.text-3xl");
            const moodElem = moodContainer.querySelector("#mood-text");
            if (percentageElem) percentageElem.innerText = `${percentage}%`;
            if (moodElem) moodElem.innerText = mood.charAt(0).toUpperCase() + mood.slice(1);
        }

        const recContainer = document.getElementById("recommended-songs-container");
        if (recContainer) {
            recContainer.innerHTML = "";
            recommendedSongs.forEach(song => {
                const songDiv = document.createElement("div");
                songDiv.className = "bg-white/10 backdrop-blur-md rounded-xl p-4 flex items-center gap-4 justify-between";
                songDiv.innerHTML = `
                    <div class="flex flex-col w-0 flex-1 min-w-0">
                        <h1 class="text-xl font-semibold truncate">${song.track_name}</h1>
                        <small class="text-white/50 text-lg truncate">${song.artists}</small>
                    </div>
                    <svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="#e3e3e3">
                        <path d="m426-330 195-125q14-9 14-25t-14-25L426-630q-15-10-30.5-1.5T380-605v250q0 18 15.5 26.5T426-330Zm54 250q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Z" />
                    </svg>
                `;
                recContainer.appendChild(songDiv);
            });
        }

    } catch (err) {
        console.error("Error predicting mood:", err);
        alert("Error predicting mood. Check console.");
    }
}
