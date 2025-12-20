"""
Curated source URLs for scenario scraping.

Organized by category with URLs to Wikipedia, Fandom wikis, and other sources
that have plot summaries and game descriptions.
"""

SOURCES = {
    # =========================================================================
    # ESCAPE ROOM / PUZZLE GAMES
    # =========================================================================
    "escape_room_games": [
        # The Room series
        "https://en.wikipedia.org/wiki/The_Room_(video_game)",
        "https://en.wikipedia.org/wiki/The_Room_Two",
        "https://en.wikipedia.org/wiki/The_Room_Three",
        "https://en.wikipedia.org/wiki/The_Room:_Old_Sins",
        # Zero Escape series
        "https://en.wikipedia.org/wiki/999:_Nine_Hours,_Nine_Persons,_Nine_Doors",
        "https://en.wikipedia.org/wiki/Virtue%27s_Last_Reward",
        "https://en.wikipedia.org/wiki/Zero_Time_Dilemma",
        # We Were Here series
        "https://en.wikipedia.org/wiki/We_Were_Here_(video_game)",
        # Other escape games
        "https://en.wikipedia.org/wiki/Escape_Academy",
        "https://en.wikipedia.org/wiki/The_Witness_(2016_video_game)",
        "https://en.wikipedia.org/wiki/Myst",
        "https://en.wikipedia.org/wiki/Riven",
        "https://en.wikipedia.org/wiki/The_Talos_Principle",
    ],

    # =========================================================================
    # MYSTERY / DETECTIVE GAMES
    # =========================================================================
    "mystery_games": [
        # Investigation games
        "https://en.wikipedia.org/wiki/Return_of_the_Obra_Dinn",
        "https://en.wikipedia.org/wiki/Her_Story_(video_game)",
        "https://en.wikipedia.org/wiki/Telling_Lies_(video_game)",
        "https://en.wikipedia.org/wiki/Immortality_(video_game)",
        "https://en.wikipedia.org/wiki/L.A._Noire",
        "https://en.wikipedia.org/wiki/Heavy_Rain",
        # Danganronpa series
        "https://en.wikipedia.org/wiki/Danganronpa:_Trigger_Happy_Havoc",
        "https://en.wikipedia.org/wiki/Danganronpa_2:_Goodbye_Despair",
        # Phoenix Wright series
        "https://en.wikipedia.org/wiki/Phoenix_Wright:_Ace_Attorney",
        "https://en.wikipedia.org/wiki/Phoenix_Wright:_Ace_Attorney_%E2%80%93_Justice_for_All",
        # Other mystery games
        "https://en.wikipedia.org/wiki/Outer_Wilds",
        "https://en.wikipedia.org/wiki/The_Vanishing_of_Ethan_Carter",
        "https://en.wikipedia.org/wiki/What_Remains_of_Edith_Finch",
        "https://en.wikipedia.org/wiki/Gone_Home",
        "https://en.wikipedia.org/wiki/Firewatch",
        "https://en.wikipedia.org/wiki/Soma_(video_game)",
    ],

    # =========================================================================
    # ADVENTURE / EXPLORATION GAMES
    # =========================================================================
    "adventure_games": [
        # Classic adventure
        "https://en.wikipedia.org/wiki/Grim_Fandango",
        "https://en.wikipedia.org/wiki/Day_of_the_Tentacle",
        "https://en.wikipedia.org/wiki/The_Secret_of_Monkey_Island",
        "https://en.wikipedia.org/wiki/Full_Throttle_(1995_video_game)",
        # Modern adventure
        "https://en.wikipedia.org/wiki/Life_Is_Strange",
        "https://en.wikipedia.org/wiki/Life_Is_Strange_2",
        "https://en.wikipedia.org/wiki/Oxenfree",
        "https://en.wikipedia.org/wiki/Night_in_the_Woods",
        "https://en.wikipedia.org/wiki/Kentucky_Route_Zero",
        "https://en.wikipedia.org/wiki/Disco_Elysium",
    ],

    # =========================================================================
    # HORROR / THRILLER GAMES
    # =========================================================================
    "horror_games": [
        "https://en.wikipedia.org/wiki/Resident_Evil_(1996_video_game)",
        "https://en.wikipedia.org/wiki/Resident_Evil_7:_Biohazard",
        "https://en.wikipedia.org/wiki/Silent_Hill_(video_game)",
        "https://en.wikipedia.org/wiki/Silent_Hill_2",
        "https://en.wikipedia.org/wiki/Amnesia:_The_Dark_Descent",
        "https://en.wikipedia.org/wiki/Outlast_(video_game)",
        "https://en.wikipedia.org/wiki/Layers_of_Fear",
        "https://en.wikipedia.org/wiki/P.T._(video_game)",
        "https://en.wikipedia.org/wiki/Alan_Wake",
        "https://en.wikipedia.org/wiki/Until_Dawn",
    ],

    # =========================================================================
    # MYSTERY / THRILLER MOVIES
    # =========================================================================
    "movie_plots": [
        # Escape room movies
        "https://en.wikipedia.org/wiki/Escape_Room_(2019_film)",
        "https://en.wikipedia.org/wiki/Escape_Room:_Tournament_of_Champions",
        # Mystery/thriller movies
        "https://en.wikipedia.org/wiki/Knives_Out",
        "https://en.wikipedia.org/wiki/Glass_Onion:_A_Knives_Out_Mystery",
        "https://en.wikipedia.org/wiki/The_Game_(1997_film)",
        "https://en.wikipedia.org/wiki/Saw_(2004_film)",
        "https://en.wikipedia.org/wiki/Cube_(1997_film)",
        "https://en.wikipedia.org/wiki/Identity_(2003_film)",
        "https://en.wikipedia.org/wiki/Clue_(film)",
        "https://en.wikipedia.org/wiki/Murder_on_the_Orient_Express_(2017_film)",
        "https://en.wikipedia.org/wiki/Death_on_the_Nile_(2022_film)",
        "https://en.wikipedia.org/wiki/The_Usual_Suspects",
        "https://en.wikipedia.org/wiki/Se7en",
        "https://en.wikipedia.org/wiki/Shutter_Island_(film)",
        "https://en.wikipedia.org/wiki/Gone_Girl_(film)",
        "https://en.wikipedia.org/wiki/Get_Out",
        "https://en.wikipedia.org/wiki/Us_(2019_film)",
        "https://en.wikipedia.org/wiki/Parasite_(2019_film)",
        # Haunted house / mystery house
        "https://en.wikipedia.org/wiki/The_Others_(2001_film)",
        "https://en.wikipedia.org/wiki/The_Sixth_Sense",
        "https://en.wikipedia.org/wiki/House_of_Wax_(2005_film)",
        "https://en.wikipedia.org/wiki/1408_(film)",
        "https://en.wikipedia.org/wiki/The_Orphanage_(film)",
    ],

    # =========================================================================
    # INTERACTIVE FICTION / VISUAL NOVELS
    # =========================================================================
    "interactive_fiction": [
        # Classic text adventures
        "https://en.wikipedia.org/wiki/Zork",
        "https://en.wikipedia.org/wiki/The_Hitchhiker%27s_Guide_to_the_Galaxy_(video_game)",
        "https://en.wikipedia.org/wiki/Planetfall",
        # Visual novels
        "https://en.wikipedia.org/wiki/Steins;Gate",
        "https://en.wikipedia.org/wiki/The_House_in_Fata_Morgana",
        "https://en.wikipedia.org/wiki/AI:_The_Somnium_Files",
        # Modern interactive fiction
        "https://en.wikipedia.org/wiki/80_Days_(2014_video_game)",
        "https://en.wikipedia.org/wiki/Sorcery!_(video_game_series)",
    ],

    # =========================================================================
    # CO-OP / MULTIPLAYER PUZZLE GAMES
    # =========================================================================
    "coop_games": [
        "https://en.wikipedia.org/wiki/Portal_2",
        "https://en.wikipedia.org/wiki/Keep_Talking_and_Nobody_Explodes",
        "https://en.wikipedia.org/wiki/It_Takes_Two_(video_game)",
        "https://en.wikipedia.org/wiki/A_Way_Out_(video_game)",
        "https://en.wikipedia.org/wiki/Brothers:_A_Tale_of_Two_Sons",
    ],

    # =========================================================================
    # SEARCH QUERIES (for web search to find more)
    # =========================================================================
    "search_queries": [
        "escape room video game plot summary",
        "mystery puzzle game premise wiki",
        "locked room mystery plot summary",
        "detective game story synopsis",
        "point and click adventure game plots",
        "cooperative puzzle game storylines",
        "visual novel mystery plots",
        "haunted house movie plot",
        "whodunit movie premises",
    ],
}


def get_all_urls() -> list:
    """Get all URLs from all categories (excluding search queries)."""
    urls = []
    for category, items in SOURCES.items():
        if category != "search_queries":
            urls.extend(items)
    return urls


def get_urls_by_category(category: str) -> list:
    """Get URLs for a specific category."""
    return SOURCES.get(category, [])


def get_categories() -> list:
    """Get list of all categories."""
    return [k for k in SOURCES.keys() if k != "search_queries"]
