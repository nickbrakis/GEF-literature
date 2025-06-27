# Enefit Dataset

**Source:** https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data

# Εισαγωγή

Το σύνολο δεδομένων αυτό είναι μέρος του διαγωνισμού Kaggle "Enefit - Predict Energy Behavior of Prosumers" σε συνεργασία με μία εσθονική εταιρεία ενέργειας, την Enefit. Σκοπός του διαγωνισμού ήταν η πρόβλεψη ηλεκτρικού φορτίου που παράγεται και καταναλώνεται από prosumers στην Εσθονία που έχουν εγκαταστήσει ηλιακά πάνελ. 

# Αρχεία Dataset

Το σύνολο δεδομένων περιλαμβάνει τα παρακάτω αρχεία:

- **train.csv** - Ιστορικές χρονοσειρές δεδομένων κατανάλωσης/παραγωγής (περιέχει την μεταβλητή **target**)
- **client.csv** - Δεδομένα πελατών
- **electricity_prices.csv** - Χρονοσειρές με τις τιμές ηλεκτρικής ενέργειας (mwh)
- **historical_weather.csv** - Ιστορικές χρονοσειρές δεδομένων καιρού
- **forecast_weather.csv** - Χρονοσειρές με προβλέψεις δεδομένων καιρού
- **gas_prices.csv** - Χρονοσειρές με τιμές αερίου
- **weather_station_to_county_mapping.csv** - Αντιστοίχιση συντεταγμένων μετεωρολογικού σταθμού με county
- **county_id_to_name_map.json** - Αντιστοίχιση county_id με όνομα county

## train.csv

- `county` - An ID code for the county.
- `is_business` - Boolean for whether or not the prosumer is a business.
- `product_type` - ID code with the following mapping of codes to contract types: `{0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}`.
- `target` - The consumption or production amount for the relevant segment for the hour. The segments are defined by the `county`, `is_business`, and `product_type`.
- `is_consumption` - Boolean for whether or not this row's target is consumption or production.
- `datetime` - The Estonian time in EET (UTC+2) / EEST (UTC+3). It describes the start of the 1-hour period on which target is given.
- `data_block_id` - All rows sharing the same `data_block_id` will be available at the same forecast time. This is a function of what information is available when forecasts are actually made, at 11 AM each morning. For example, if the forecast weather `data_block_id` for predictins made on October 31st is 100 then the historic weather `data_block_id` for October 31st will be 101 as the historic weather data is only actually available the next day.
- `row_id` - A unique identifier for the row.
- `prediction_unit_id` - A unique identifier for the `county`, `is_business`, and `product_type` combination. *New prediction units can appear or disappear in the test set.*

# Μεταβλητή Στόχος

Η μεταβλητή στόχος της πρόβλεψης είναι η target και αντιπροσωπεύει τη ποσότητα κατανάλωσης ή παραγωγής ηλεκτρικής ενέργειας.