async function predictMood() {
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

        // Map moods to emojis
        const moodEmojiMap = {
            happy: "😊",
            sad: "😢",
            calm: "😌",
            energetic: "🥳",
            neutral: "😐"
        };
        const emoji = moodEmojiMap[mood] || "🎵";

        // Update ApexCharts radial chart
        chart.updateOptions({
            series: [percentage],
            labels: [emoji],
        });

        // Update mood text and percentage next to chart
        const moodContainer = document.querySelector("#demo .col-span-1.row-span-1 div.flex.flex-col.items-center.relative");
        if (moodContainer) {
            const percentageElem = moodContainer.querySelector("h1.text-3xl");
            const moodElem = moodContainer.querySelector("#mood-text");
            if (percentageElem) percentageElem.innerText = `${percentage}%`;
            if (moodElem) moodElem.innerText = mood;
        }

    } catch (err) {
        console.error("Error predicting mood:", err);
        alert("Error predicting mood. Check console.");
    }
}
