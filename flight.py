import folium
from folium.plugins import MarkerCluster
import requests

def get_real_time_flights():
    """
    Fetch real-time flight data using OpenSky Network API.
    Returns a list of flights with callsign, longitude, latitude, altitude, and country.
    """
    try:
        url = "https://opensky-network.org/api/states/all"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            flights = []
            for state in data['states']:
                if state[5] is not None and state[6] is not None:
                    flights.append({
                        'callsign': state[1].strip() if state[1] else 'Unknown',
                        'longitude': state[5],
                        'latitude': state[6],
                        'altitude': state[7] if state[7] is not None else 0,
                        'country': state[2]
                    })
            return flights
        else:
            print("Failed to fetch flight data")
            return []
    except Exception as e:
        print(f"Error fetching flight data: {e}")
        return []

def create_global_map():
    """
    Creates a static Folium map with:
    - A custom TileLayer that does not repeat the world.
    - Real-time flight data displayed using MarkerCluster.
    - Popular aurora viewing locations.
    """
    # Create the base map without default tiles and enable max_bounds
    # to restrict panning to a single world image.
    m = folium.Map(location=[30, 0], zoom_start=3, tiles=None, max_bounds=True)

    # Add a TileLayer with the no_wrap property so that the map doesn't repeat.
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='OpenStreetMap',
        max_zoom=18,
        no_wrap=True
    ).add_to(m)

    # Feature Group: Real-time Flight Data with Marker Clustering
    flights_group = folium.FeatureGroup(name="Real-time Flights")
    flight_cluster = MarkerCluster().add_to(flights_group)
    
    flights = get_real_time_flights()
    for flight in flights:
        folium.Marker(
            location=[flight['latitude'], flight['longitude']],
            popup=(f"Callsign: {flight['callsign']}<br>"
                  f"Country: {flight['country']}<br>"
                  f"Altitude: {flight['altitude']} m"),
            icon=folium.Icon(color='blue', icon='plane', prefix='fa')
        ).add_to(flight_cluster)
    
    flights_group.add_to(m)

    # Feature Group: Popular Aurora Viewing Locations
    locations_group = folium.FeatureGroup(name="Aurora Viewing Spots")
    
    aurora_locations = [
        {"name": "Tromsø, Norway", "location": [69.6, 18.9], "description": "Prime Northern Lights Viewing"},
        {"name": "Fairbanks, Alaska", "location": [64.8, -147.7], "description": "Aurora Borealis Hotspot"},
        {"name": "Yellowknife, Canada", "location": [62.4, -114.4], "description": "Clear Skies for Aurora"},
        {"name": "Reykjavik, Iceland", "location": [64.1, -21.9], "description": "Aurora Capital"}
    ]
    
    for loc in aurora_locations:
        folium.Marker(
            location=loc['location'],
            popup=f"{loc['name']}<br>{loc['description']}",
            tooltip=loc['name'],
            icon=folium.Icon(color='green', icon='eye', prefix='fa')
        ).add_to(locations_group)
    
    locations_group.add_to(m)

    # Add layer control to toggle flight data and aurora viewing locations on/off
    folium.LayerControl().add_to(m)
    
    return m

# Create the map and save it to an HTML file
global_map = create_global_map()
global_map.save('real_time_global_map.html')

global_map