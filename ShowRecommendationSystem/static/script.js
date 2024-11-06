const showGrid = document.getElementById('show-grid');
const recommendationsGrid = document.getElementById('recommendations');
const apiUrl = 'http://127.0.0.1:8000';

// Function to fetch and display all shows
async function loadShows() {
    try {
        const response = await fetch(`${apiUrl}/shows`);
        const shows = await response.json();
        console.log(shows)
        showGrid.innerHTML = '';
        shows.forEach(show => {
            const showDiv = document.createElement('div');
            showDiv.classList.add('show-item');
            const picURL = `https://image.tmdb.org/t/p/w500${show.poster_path}`;
            console.log(picURL);
            showDiv.style.backgroundImage = `url(${picURL})`;
            showDiv.innerText = show.name;
            showDiv.onclick = () => getRecommendations(show.name);
            showGrid.appendChild(showDiv);
        });
    } catch (error) {
        console.error('Error loading shows:', error);
        showGrid.innerHTML = '<p>Error fetching recommendations. Please try again.</p>';
    }
}

// Function to get and display recommendations
async function getRecommendations(title) {
    try {
        console.log(title)
        const response = await fetch(`${apiUrl}/recommend?title=${encodeURIComponent(title)}`);
        console.log(response)
        const recommendedShows = await response.json();
        console.log(recommendedShows)
        // Check if the response data is an array
        if (!Array.isArray(recommendedShows)) {
            throw new Error("Expected an array of recommended shows.");
        }
        recommendationsGrid.innerHTML = '';
        recommendedShows.forEach(show => {
            try {
                const recDiv = document.createElement('div');
                recDiv.classList.add('show-item');
                const picURL = `https://image.tmdb.org/t/p/w500${show.poster_path}`;
                recDiv.style.backgroundImage = `url(${picURL})`;
                recDiv.innerText = show.name;
                recommendationsGrid.appendChild(recDiv);
            } catch (innerError) {
                console.error("Error adding show to UI:", innerError);
            }
        });
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        recommendationsGrid.innerHTML = '<p>Error fetching recommendations. Please try again.</p>';
    }
}

// Load all shows on page load
window.onload = loadShows;
