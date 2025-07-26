// Video Search Platform - JavaScript Functionality

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchType = document.getElementById('searchType');
const autocompleteDropdown = document.getElementById('autocompleteDropdown');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsContainer = document.getElementById('resultsContainer');
const resultsList = document.getElementById('resultsList');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const noResults = document.getElementById('noResults');
const searchStats = document.getElementById('searchStats');
const similarModal = document.getElementById('similarModal');
const similarVideosList = document.getElementById('similarVideosList');

// State
let autocompleteTimeout;
let selectedAutocompleteIndex = -1;
let currentAutocompleteItems = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateSearchStats();
});

// Event Listeners
function initializeEventListeners() {
    searchInput.addEventListener('input', handleSearchInput);
    searchInput.addEventListener('keydown', handleKeydown);
    searchInput.addEventListener('focus', showAutocomplete);
    searchInput.addEventListener('blur', hideAutocompleteDelayed);
    searchBtn.addEventListener('click', handleSearch);
    searchType.addEventListener('change', handleSearchTypeChange);
    
    // Click outside to close autocomplete
    document.addEventListener('click', function(event) {
        if (!autocompleteDropdown.contains(event.target) && event.target !== searchInput) {
            hideAutocomplete();
        }
    });
}

// Search Input Handler
function handleSearchInput(event) {
    const query = event.target.value.trim();
    
    if (query.length >= 2) {
        clearTimeout(autocompleteTimeout);
        autocompleteTimeout = setTimeout(() => {
            fetchAutocomplete(query);
        }, 300);
    } else {
        hideAutocomplete();
    }
}

// Keyboard Navigation
function handleKeydown(event) {
    const items = autocompleteDropdown.querySelectorAll('.autocomplete-item');
    
    switch (event.key) {
        case 'ArrowDown':
            event.preventDefault();
            selectedAutocompleteIndex = Math.min(selectedAutocompleteIndex + 1, items.length - 1);
            updateAutocompleteSelection();
            break;
            
        case 'ArrowUp':
            event.preventDefault();
            selectedAutocompleteIndex = Math.max(selectedAutocompleteIndex - 1, -1);
            updateAutocompleteSelection();
            break;
            
        case 'Enter':
            event.preventDefault();
            if (selectedAutocompleteIndex >= 0 && items[selectedAutocompleteIndex]) {
                const selectedText = currentAutocompleteItems[selectedAutocompleteIndex].text;
                searchInput.value = selectedText;
                hideAutocomplete();
                performSearch(selectedText);
            } else {
                handleSearch();
            }
            break;
            
        case 'Escape':
            hideAutocomplete();
            break;
    }
}

// Search Functions
function handleSearch() {
    const query = searchInput.value.trim();
    if (query) {
        performSearch(query);
    }
}

function performSearch(query, searchTypeOverride = null) {
    if (typeof query === 'string') {
        searchInput.value = query;
    }
    
    const finalQuery = searchInput.value.trim();
    const type = searchTypeOverride || searchType.value;
    
    if (!finalQuery) {
        showNoResults('Please enter a search term');
        return;
    }
    
    showLoading();
    hideAutocomplete();
    
    const params = new URLSearchParams({
        q: finalQuery,
        type: type,
        limit: '20'
    });
    
    fetch(`/api/search?${params}`)
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.results && data.results.length > 0) {
                displayResults(data.results, data.query, data.search_type);
            } else {
                showNoResults(`No videos found for "${finalQuery}"`);
            }
            updateSearchStats(`Last search: "${finalQuery}" (${data.count || 0} results)`);
        })
        .catch(error => {
            console.error('Search error:', error);
            hideLoading();
            showNoResults('Search failed. Please try again.');
        });
}

// Autocomplete Functions
function fetchAutocomplete(query) {
    const params = new URLSearchParams({
        q: query,
        limit: '8'
    });
    
    fetch(`/api/autocomplete?${params}`)
        .then(response => response.json())
        .then(data => {
            displayAutocomplete(data.suggestions || []);
        })
        .catch(error => {
            console.error('Autocomplete error:', error);
        });
}

function displayAutocomplete(suggestions) {
    currentAutocompleteItems = suggestions;
    selectedAutocompleteIndex = -1;
    
    if (suggestions.length === 0) {
        hideAutocomplete();
        return;
    }
    
    autocompleteDropdown.innerHTML = suggestions.map((suggestion, index) => `
        <div class="autocomplete-item" onclick="selectAutocomplete('${suggestion.text.replace(/'/g, '\\\'')}')" data-index="${index}">
            <div class="autocomplete-text">${suggestion.text}</div>
            <div class="autocomplete-category">${suggestion.category}</div>
        </div>
    `).join('');
    
    showAutocomplete();
}

function selectAutocomplete(text) {
    searchInput.value = text;
    hideAutocomplete();
    performSearch(text);
}

function updateAutocompleteSelection() {
    const items = autocompleteDropdown.querySelectorAll('.autocomplete-item');
    items.forEach((item, index) => {
        item.classList.toggle('selected', index === selectedAutocompleteIndex);
    });
}

function showAutocomplete() {
    if (currentAutocompleteItems.length > 0) {
        autocompleteDropdown.style.display = 'block';
    }
}

function hideAutocomplete() {
    autocompleteDropdown.style.display = 'none';
    selectedAutocompleteIndex = -1;
}

function hideAutocompleteDelayed() {
    setTimeout(hideAutocomplete, 150);
}

// Results Display Functions
function displayResults(results, query, searchType) {
    resultsTitle.textContent = `Search Results for "${query}"`;
    resultsCount.textContent = `${results.length} video${results.length !== 1 ? 's' : ''} found`;
    
    resultsList.innerHTML = results.map(video => createVideoCard(video)).join('');
    
    showResults();
    
    // Add fade-in animation
    resultsList.classList.add('fade-in');
}

function createVideoCard(video) {
    const genreTags = video.genre.map(genre => 
        `<span class="genre-tag">${genre}</span>`
    ).join('');
    
    const actorTags = video.actors.slice(0, 3).map(actor => 
        `<span class="actor-tag">${actor}</span>`
    ).join('');
    
    const moreActors = video.actors.length > 3 ? 
        `<span class="actor-tag">+${video.actors.length - 3} more</span>` : '';
    
    return `
        <div class="video-card slide-in">
            <div class="match-type">${video.match_type}</div>
            <h3 class="video-title">${video.title}</h3>
            <div class="video-meta">
                <span>${video.year}</span>
                <span class="video-rating">★ ${video.rating}</span>
                <span>Score: ${video.score.toFixed(2)}</span>
            </div>
            <div class="video-genres">
                <strong>Genres:</strong> ${genreTags}
            </div>
            <div class="video-actors">
                <strong>Cast:</strong> ${actorTags}${moreActors}
            </div>
            <div class="video-description">
                ${video.description}
            </div>
            <div class="video-actions">
                <button class="action-btn primary-btn" onclick="findSimilarVideos(${video.id})">
                    Find Similar
                </button>
                <button class="action-btn secondary-btn" onclick="searchByActor('${video.actors[0]}')">
                    More by ${video.actors[0].split(' ')[0]}
                </button>
            </div>
        </div>
    `;
}

// Similar Videos Functions
function findSimilarVideos(videoId) {
    const params = new URLSearchParams({
        id: videoId,
        limit: '8'
    });
    
    fetch(`/api/similar?${params}`)
        .then(response => response.json())
        .then(data => {
            if (data.similar_videos && data.similar_videos.length > 0) {
                displaySimilarVideos(data.similar_videos);
            } else {
                alert('No similar videos found');
            }
        })
        .catch(error => {
            console.error('Similar videos error:', error);
            alert('Failed to find similar videos');
        });
}

function displaySimilarVideos(videos) {
    similarVideosList.innerHTML = videos.map(video => `
        <div class="similar-video-item">
            <div class="similarity-score">
                ${(video.similarity_score * 100).toFixed(0)}% match
            </div>
            <h4>${video.title} (${video.year})</h4>
            <div class="video-genres">
                ${video.genre.map(g => `<span class="genre-tag">${g}</span>`).join('')}
            </div>
            <div style="margin-top: 10px;">
                <strong>Rating:</strong> ★ ${video.rating}
            </div>
            <button class="action-btn primary-btn" style="margin-top: 10px;" 
                    onclick="searchByTitle('${video.title}'); closeSimilarModal();">
                View Details
            </button>
        </div>
    `).join('');
    
    similarModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeSimilarModal() {
    similarModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Quick Search Functions
function searchByActor(actorName) {
    searchInput.value = actorName;
    performSearch(actorName, 'actor');
}

function searchByTitle(title) {
    searchInput.value = title;
    performSearch(title, 'title');
}

function searchByGenre(genre) {
    searchInput.value = genre;
    performSearch(genre, 'genre');
}

// UI State Functions
function showLoading() {
    loadingIndicator.style.display = 'block';
    resultsContainer.style.display = 'none';
    noResults.style.display = 'none';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
}

function showResults() {
    hideLoading();
    resultsContainer.style.display = 'block';
    noResults.style.display = 'none';
}

function showNoResults(message) {
    hideLoading();
    resultsContainer.style.display = 'none';
    noResults.style.display = 'block';
    
    const noResultsElement = document.querySelector('.no-results h3');
    if (noResultsElement) {
        noResultsElement.textContent = message;
    }
}

function handleSearchTypeChange() {
    const currentQuery = searchInput.value.trim();
    if (currentQuery) {
        updateSearchStats(`Search type changed to: ${searchType.options[searchType.selectedIndex].text}`);
    }
}

function updateSearchStats(message = '') {
    if (message) {
        searchStats.textContent = message;
    } else {
        searchStats.textContent = 'Enter a search term to begin...';
    }
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Demo Suggestions
function showDemoSuggestions() {
    const suggestions = [
        { text: 'The Matrix', type: 'title' },
        { text: 'Tom Hanks', type: 'actor' },
        { text: 'Action', type: 'genre' },
        { text: '1994', type: 'year' },
        { text: 'Christopher Nolan', type: 'actor' },
        { text: 'Crime', type: 'genre' }
    ];
    
    return suggestions;
}

// Error Handling
window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
});

// Export functions for global access
window.performSearch = performSearch;
window.findSimilarVideos = findSimilarVideos;
window.closeSimilarModal = closeSimilarModal;
window.searchByActor = searchByActor;
window.searchByTitle = searchByTitle;
window.searchByGenre = searchByGenre; 