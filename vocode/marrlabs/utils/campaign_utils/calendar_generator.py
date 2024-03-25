import argparse
import datetime
import random
import json

def generate_available_schedule(available_slots, unavailability_odds):
    """ Randomly select available time slots from a 
    list of slots. """
    return [slot for slot in available_slots 
            if random.choice([True] + [False] * unavailability_odds)]

def generate_weekday_availability(start_date, unavailability_odds, weeks=2):
    """ Generate a calendar with random available slots for 
    weekdays over a specified number of weeks. """
    time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                  "13:00-14:00", "14:00-15:00", "15:00-16:00", 
                  "16:00-17:00"]
    availability_calendar = {"availability": [], "unavailability_odds":unavailability_odds}

    # Calculate end date
    end_date = start_date + datetime.timedelta(weeks=weeks)

    # Generate dates for the weekdays
    current_date = start_date
    while current_date < end_date:
        # Weekday (0-4 are Monday-Friday)
        if current_date.weekday() < 5:
            day_of_week = current_date.strftime("%A")  # Gets the full weekday name
            availability_calendar["availability"].append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day": day_of_week, 
                "time_slots": generate_available_schedule(time_slots, unavailability_odds)
            })
        current_date += datetime.timedelta(days=1)

    return availability_calendar


def run_argparse():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="""Generate 
                                     a calendar with random available 
                                     time slots for weekdays for the next 
                                     two weeks.""")
    
    parser.add_argument("--start-date", help="The start date in YYYY-MM-DD format", required=True)
    parser.add_argument("--unavailability-odds", type=int, 
                        help="Odds-to-one for unavailability", default=1)

    # Parse arguments
    args = parser.parse_args()
    return args

def main():
    args = run_argparse()
    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    availability_odds = args.unavailability_odds

    # Generate availability calendar
    availability_calendar = generate_weekday_availability(start_date, availability_odds)
    availability_calendar_json = json.dumps(availability_calendar, indent=4)
    print(availability_calendar)

if __name__ == "__main__":
    main()