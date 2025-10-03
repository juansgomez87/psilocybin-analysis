"""
Interactive Psilocybin Playlist Visualizer

Creates an interactive HTML file with dropdown menus and buttons for filtering and selection.
"""

import logging
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import glob
import json
import argparse
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """Creates an interactive HTML visualizer with dropdown controls."""

    def __init__(self, base_path='.'):
        """Initialize visualizer with data."""
        self.base_path = base_path
        self.playlists = ['chacruna_baldwin', 'chacruna_kelan_thomas2', 'compass_v2',
                     'copenhagen', 'imperial1', 'imperial2', 'jh_classical', 'jh_overtone']
        self.polar_feats = ['Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live']
        
        # Load playlist metadata
        self.load_playlist_metadata()
        
        # Load processed data
        self.data_dir = 'data'
        self.load_processed_data()

    def load_playlist_metadata(self):
        """Load metadata from playlist CSV files."""
        full_data_file = 'data/full_data.csv'
        if os.path.exists(full_data_file):
            self.csv = pd.read_csv(full_data_file, index_col=0)
            logger.info(f"Loaded playlist metadata from: {full_data_file}")


    def load_processed_data(self):
        """Load processed feature data."""
        self.all_dfs = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not self.all_dfs:
            logger.error(f"No CSV files found in {self.data_dir}")
            self.data = pd.DataFrame()
            return
            
        # Load the first available file by default
        self.data = pd.read_csv(self.all_dfs[0], index_col=0)
        logger.info(f"Loaded data from: {self.all_dfs[0]}")

    def create_interactive_html(self, output_file='interactive_psilocybin_visualizer.html'):
        """Create an interactive HTML file with dropdown controls."""
        
        # Get available data files and load them
        available_data = []
        data_dict = {}
        emb_types = ['compare_lld', 'compare_lld_mean']
        
        for emb_type in emb_types:
            data_file = os.path.join(self.data_dir, f'df_{emb_type}.csv')
            if os.path.exists(data_file):
                data_key = emb_type
                available_data.append(data_key)
                # Load the data and convert to JSON
                df = pd.read_csv(data_file, index_col=0)
                data_dict[data_key] = df.to_dict('records')
        
        # Convert song metadata to JSON
        song_metadata_json = json.dumps(self.csv.to_dict('records'), default=str)
        data_json = json.dumps(data_dict, default=str)
        
        # Generate HTML components separately to avoid f-string conflicts
        html_template = self._get_html_template()
        css_styles = self._get_css_styles()
        javascript_code = self._get_javascript_code()
        
        # Create the complete HTML content
        html_content = html_template.format(
            options=self._generate_options(available_data),
            playlist_options=self._generate_playlist_options(),
            song_metadata_json=song_metadata_json,
            data_json=data_json,
            css_styles=css_styles,
            javascript_code=javascript_code
        )
        
        # Write the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Validate the generated HTML
        self._validate_html(html_content)
        
        logger.info(f"Interactive HTML visualizer created: {output_file}")
        return output_file

    def _generate_options(self, available_data):
        """Generate HTML options for embedding types."""
        options = []
        for data_type in available_data:
            display_name = data_type.replace('_', ' ').title()
            options.append(f'<option value="{data_type}">{display_name}</option>')
        return '\n                    '.join(options)

    def _generate_playlist_options(self):
        """Generate HTML options for playlists."""
        options = []
        for playlist in self.playlists:
            display_name = playlist.replace('_', ' ').title()
            options.append(f'<option value="{playlist}">{display_name}</option>')
        return '\n                    '.join(options)

    def _get_html_template(self):
        """Get the HTML template without f-string conflicts."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Interactive Psilocybin Playlist Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <meta charset="UTF-8">
    <style>
{css_styles}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Interactive Psilocybin Playlist Visualizer</h1>
            <p>Explore music used in psilocybin therapy through interactive visualizations</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="embType">Embedding Type:</label>
                <select id="embType">
                    {options}
                </select>
            </div>
            
            <div class="control-group">
                <label for="colorBy">Color By:</label>
                <select id="colorBy">
                    <option value="playlist">Playlist</option>
                    <option value="phase">Phase</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="playlistFilter">Filter by Playlist:</label>
                <select id="playlistFilter">
                    <option value="all">All Playlists</option>
                    {playlist_options}
                </select>
            </div>
            
            <div class="control-group">
                <label for="artistFilter">Filter by Artist:</label>
                <select id="artistFilter">
                    <option value="all">All Artists</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="updatePlot()">Update Plot</button>
            </div>
            
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="resetFilters()">Reset Filters</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="plot-container">
                <div id="loading" class="loading">Loading plot...</div>
                <div id="plot"></div>
            </div>
            
            <div class="sidebar">
                <div class="stats">
                    <h4>Dataset Statistics</h4>
                    <div id="stats"></div>
                </div>
                
                <div id="songInfo" class="song-info">
                    <h3>Song Information</h3>
                    <div id="songDetails"></div>
                    <div id="spotifyEmbed" class="spotify-embed"></div>
                    <div id="polarPlot" class="polar-plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentData = null;
        let allData = null;
        let songMetadata = {song_metadata_json};
        let dataDict = {data_json};
        
        {javascript_code}
    </script>
</body>
</html>"""

    def _get_css_styles(self):
        """Get CSS styles as a string."""
        return """        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            min-width: 200px;
        }
        .control-group label {
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }
        .control-group select, .control-group button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .control-group select:focus, .control-group button:focus {
            outline: none;
            border-color: #007bff;
        }
        .control-group button {
            background-color: #007bff;
            color: white;
            cursor: pointer;
            border: none;
        }
        .control-group button:hover {
            background-color: #0056b3;
        }
        .plot-container {
            margin-top: 20px;
        }
        .song-info {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        .song-info h3 {
            margin-top: 0;
            color: #333;
        }
        .song-info p {
            margin: 5px 0;
            color: #666;
        }
        .spotify-embed {
            margin: 20px 0;
        }
        .polar-plot {
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .main-content {
            display: flex;
            gap: 20px;
        }
        .plot-container {
            flex: 2;
            margin-top: 20px;
        }
        .sidebar {
            flex: 1;
            margin-top: 20px;
        }
        .stats {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats h4 {
            margin-top: 0;
            color: #333;
        }
        .stats p {
            margin: 5px 0;
            color: #666;
        }"""

    def _get_javascript_code(self):
        """Get JavaScript code as a string."""
        return """        // Initialize the visualization
        document.addEventListener('DOMContentLoaded', function() {
            // Add a small delay to ensure all elements are fully loaded
            setTimeout(function() {
                loadData();
                setupEventListeners();
                updateStats();
            }, 100);
        });
        
        function setupEventListeners() {
            // Update artist filter when playlist filter changes
            document.getElementById('playlistFilter').addEventListener('change', updateArtistFilter);
            
            // Update plot when embedding type or color changes
            document.getElementById('embType').addEventListener('change', updatePlot);
            document.getElementById('colorBy').addEventListener('change', updatePlot);
        }
        
        function loadData() {
            const embType = document.getElementById('embType').value;
            allData = dataDict[embType] || [];
            currentData = allData;
            updateArtistFilter();
            updatePlot();
            updateStats();
        }
        
        function updateStats() {
            if (!allData) return;
            
            const stats = document.getElementById('stats');
            if (!stats) {
                console.error('Stats element not found');
                return;
            }
            
            const totalSongs = allData.length;
            const playlists = [...new Set(allData.map(row => row.playlist))];
            const artists = [...new Set(allData.map(row => row.artist))];
            
            stats.innerHTML = 
                '<p><strong>Total Songs:</strong> ' + totalSongs + '</p>' +
                '<p><strong>Playlists:</strong> ' + playlists.length + '</p>' +
                '<p><strong>Artists:</strong> ' + artists.length + '</p>' +
                '<p><strong>Current View:</strong> <span id="currentStats">' + totalSongs + ' songs</span></p>';
        }
        
        function updateArtistFilter() {
            const playlistFilterElement = document.getElementById('playlistFilter');
            const artistSelect = document.getElementById('artistFilter');
            
            if (!playlistFilterElement || !artistSelect) {
                console.error('Required filter elements not found');
                return;
            }
            
            const playlistFilter = playlistFilterElement.value;
            
            // Clear current options
            artistSelect.innerHTML = '<option value="all">All Artists</option>';
            
            if (!allData || playlistFilter === 'all') return;
            
            // Get unique artists for the selected playlist
            const artists = [...new Set(allData
                .filter(row => row.playlist === playlistFilter)
                .map(row => row.artist)
                .filter(artist => artist))];
            
            artists.forEach(artist => {
                const option = document.createElement('option');
                option.value = artist;
                option.textContent = artist;
                artistSelect.appendChild(option);
            });
        }
        
        function updatePlot() {
            if (!allData) return;
            
            // Check if all required elements exist
            const embTypeElement = document.getElementById('embType');
            const colorByElement = document.getElementById('colorBy');
            const playlistFilterElement = document.getElementById('playlistFilter');
            const artistFilterElement = document.getElementById('artistFilter');
            const currentStatsElement = document.getElementById('currentStats');
            const plotElement = document.getElementById('plot');
            const loadingElement = document.getElementById('loading');
            
            if (!embTypeElement || !colorByElement || !playlistFilterElement || !artistFilterElement) {
                console.error('Required form elements not found');
                return;
            }
            
            const embType = embTypeElement.value;
            const colorBy = colorByElement.value;
            const playlistFilter = playlistFilterElement.value;
            const artistFilter = artistFilterElement.value;
            
            // Filter data
            let filteredData = allData;
            
            if (playlistFilter !== 'all') {
                filteredData = filteredData.filter(row => row.playlist === playlistFilter);
            }
            
            if (artistFilter !== 'all') {
                filteredData = filteredData.filter(row => row.artist === artistFilter);
            }
            
            currentData = filteredData;
            
            // Update stats
            if (currentStatsElement) {
                currentStatsElement.textContent = filteredData.length + ' songs';
            }
            
            // Group by color variable
            const groups = {};
            filteredData.forEach(row => {
                const group = row[colorBy] || 'Unknown';
                if (!groups[group]) {
                    groups[group] = [];
                }
                groups[group].push(row);
            });
            
            // Create traces
            const traces = [];
            Object.keys(groups).forEach(group => {
                const groupData = groups[group];
                traces.push({
                    x: groupData.map(row => parseFloat(row.umap_x)),
                    y: groupData.map(row => parseFloat(row.umap_y)),
                    mode: 'markers',
                    type: 'scattergl',
                    name: group,
                    marker: {
                        size: 6,
                        opacity: 0.7,
                        line: { width: 1 }
                    },
                    text: groupData.map(row => 
                        'Playlist: ' + row.playlist + '<br>Artist: ' + row.artist + '<br>Track: ' + row.song + '<br>Spotify ID: ' + row.spotify_id
                    ),
                    hoverinfo: 'text'
                });
            });
            
            const layout = {
                title: embType.toUpperCase() + ' - ' + colorBy.charAt(0).toUpperCase() + colorBy.slice(1),
                xaxis: { title: 'UMAP Dimension 1' },
                yaxis: { title: 'UMAP Dimension 2' },
                margin: { l: 0, r: 0, b: 0, t: 40 },
                legend: { x: 0, y: 0, orientation: 'h' },
                template: 'plotly_white',
                height: 600,
                width: 800
            };
            
            if (plotElement) {
                Plotly.newPlot('plot', traces, layout, {responsive: true});
                
                // Add click event
                plotElement.addEventListener('plotly_click', function(data) {
                const point = data.points[0];
                const text = point.text;
                const spotifyIdMatch = text.match(/Spotify ID: ([a-zA-Z0-9]+)/);
                const playlistMatch = text.match(/Playlist: ([a-zA-Z0-9_]+)/);
                
                if (spotifyIdMatch && playlistMatch) {
                    const spotifyId = spotifyIdMatch[1];
                    const playlist = playlistMatch[1];
                    showSongInfo(spotifyId, playlist);
                }
            });
            } else {
                console.error('Plot element not found');
            }
            
            if (loadingElement) {
                loadingElement.style.display = 'none';
            }
        }
        
        function resetFilters() {
            const playlistFilterElement = document.getElementById('playlistFilter');
            const artistFilterElement = document.getElementById('artistFilter');
            
            if (playlistFilterElement && artistFilterElement) {
                playlistFilterElement.value = 'all';
                artistFilterElement.value = 'all';
                updateArtistFilter();
                updatePlot();
            } else {
                console.error('Filter elements not found for reset');
            }
        }
        
        function showSongInfo(spotifyId, playlist) {
            // Find song in metadata
            const songData = songMetadata.find(song => 
                song['Spotify Track Id'] === spotifyId && song.playlist === playlist
            );
            
            const songInfo = document.getElementById('songInfo');
            const songDetails = document.getElementById('songDetails');
            const spotifyEmbed = document.getElementById('spotifyEmbed');
            const polarPlot = document.getElementById('polarPlot');
            
            if (!songInfo || !songDetails || !spotifyEmbed || !polarPlot) {
                console.error('Required song info elements not found');
                return;
            }
            
            if (songData) {
                songDetails.innerHTML = 
                    '<p><strong>Title:</strong> ' + songData.Song + ' (' + songData['Album Date'] + ')</p>' +
                    '<p><strong>Artist:</strong> ' + songData.Artist + '</p>' +
                    '<p><strong>Album:</strong> ' + songData.Album + '</p>' +
                    '<p><strong>Time:</strong> ' + songData.Time + '</p>' +
                    '<p><strong>Genres:</strong> ' + songData.Genres + '</p>' +
                    '<p><strong>Key:</strong> ' + songData.Key + '</p>' +
                    '<p><strong>Time Signature:</strong> ' + songData['Time Signature'] + '/4</p>' +
                    '<p><strong>Camelot:</strong> ' + songData.Camelot + '</p>' +
                    '<p><strong>Loudness:</strong> ' + songData['Loud (Db)'] + ' dB</p>';
                
                // Create polar plot with Spotify features
                const polarFeatures = ['Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live'];
                const values = polarFeatures.map(feat => parseFloat(songData[feat]));
                
                const polarTrace = {
                    r: values,
                    theta: polarFeatures,
                    fill: 'toself',
                    name: songData.Song,
                    type: 'scatterpolar',
                    marker: { color: 'rgba(0, 123, 255, 0.7)' }
                };
                
                const polarLayout = {
                    polar: {
                        radialaxis: {
                            visible: true,
                            range: [0, 100]
                        }
                    },
                    showlegend: false,
                    title: 'Spotify Features',
                    height: 300,
                    width: 300
                };
                
                Plotly.newPlot('polarPlot', [polarTrace], polarLayout);
                
            } else {
                songDetails.innerHTML = 
                    '<p><strong>Spotify ID:</strong> ' + spotifyId + '</p>' +
                    '<p><strong>Playlist:</strong> ' + playlist + '</p>' +
                    '<p><em>Song details not found in metadata...</em></p>';
                polarPlot.innerHTML = '<p><em>Polar plot not available...</em></p>';
            }
            
            spotifyEmbed.innerHTML = 
                                 '<iframe src="https://open.spotify.com/embed/track/' + spotifyId + '" ' +
                 'width="300" height="100" frameborder="0" ' +
                 'allowtransparency="true" allow="encrypted-media"></iframe>';
            
            songInfo.style.display = 'block';
        }"""

    def _validate_html(self, html_content):
        """Validate that the generated HTML is syntactically correct."""
        try:
            # Check for basic HTML structure
            if '<!DOCTYPE html>' not in html_content:
                logger.warning("Missing DOCTYPE declaration")
            
            if '<html>' not in html_content or '</html>' not in html_content:
                logger.warning("Missing HTML tags")
            
            if '<head>' not in html_content or '</head>' not in html_content:
                logger.warning("Missing HEAD tags")
            
            if '<body>' not in html_content or '</body>' not in html_content:
                logger.warning("Missing BODY tags")
            
            # Check for JavaScript syntax issues
            if '{{' in html_content or '}}' in html_content:
                logger.warning("Found unescaped curly braces in HTML content")
            
            # Check for common JavaScript errors
            js_start = html_content.find('<script>')
            js_end = html_content.find('</script>')
            if js_start != -1 and js_end != -1:
                js_code = html_content[js_start:js_end]
                # Check for balanced parentheses and braces
                if js_code.count('(') != js_code.count(')'):
                    logger.warning("Unbalanced parentheses in JavaScript")
                if js_code.count('{') != js_code.count('}'):
                    logger.warning("Unbalanced braces in JavaScript")
            
            logger.info("HTML validation completed successfully")
            
        except Exception as e:
            logger.error(f"HTML validation failed: {e}")


def main():
    """Main function to create the interactive visualizer."""
    parser = argparse.ArgumentParser(description='Create interactive HTML psilocybin playlist visualizer')
    parser.add_argument('--base-path', type=str, default='/Users/jsgomezc/Data/psilocybin',
                       help='Base path for data')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing processed data files')
    parser.add_argument('--output-file', type=str, default='interactive_psilocybin_visualizer.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = InteractiveVisualizer(base_path=args.base_path)
    visualizer.data_dir = args.data_dir
    
    # Create interactive HTML
    output_file = visualizer.create_interactive_html(args.output_file)
    
    print(f"Interactive visualizer created: {output_file}")
    print("Open this file in your web browser to use the interactive features.")


if __name__ == "__main__":
    main()
