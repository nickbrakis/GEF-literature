[Targeted demand response for flexible energy communities using
clustering techniques](https://cooked-flier-beb.notion.site/Targeted-demand-response-for-flexible-energy-communities-using-clustering-techniques-218065be0c0380398ecbccd3900cc847)

Στα πλαίσια της παρούσας μελέτης εξετάζεται ο σχεδιασμός προγραμμάτων Απόκρισης-Ζήτησης (Α/Ζ) με χρήση τεχνικών συσταδοποίησης καταναλωτών και παραγωγών (prosumers). Η συσταδοποίηση που ερευνάται αφορά ενεργειακή κοινότητα στην Ιταλία και για την αξιολόγηση της εκτός από τις τυπικές εισάγεται μία νέα αποτελεσματική μετρική, Peak Performance Score (PPS). 

Αρχικά αναλύονται συνοπτικά και κατατοπιστικά ζητήματα προγραμμάτων Απόκρισης Ζήτησης, εστιάζοντας στην απόκριση βάση τιμής. Τρία βασικά προγράμματα αναγνωρίζονται σε αυτή την κατηγορία: 

1.  Χρονικά Διαμορφωμένη Τιμολόγηση (TOU): Οι ώρες τις ημέρας χωρίζονται σε διακριτά διαστήματα. Οι ώρες αιχμής έχουν υψηλότερη τιμή.
2. Τιμολόγηση Κρίσιμων Αιχμών (CPP): Τα διαστήματα των ωρών με υψηλότερη τιμή δεν είναι προκαθορισμένα και αλλάζουν ανάλογα με προβλέψεις και γεγονότα. 
3. Τιμολόγηση Πραγματικού Χρόνου (RTP): Η τιμή αλλάζει δυναμικά τουλάχιστον 1 ώρα πριν ανακοινωθεί στους καταναλωτές. 

Στην συνέχεια αναλύεται η σημασία του όρου της εντροπίας στα ζητούμενα της Α/Ζ. Σύμφωνα με τον όρο της “μεταβλητότητας φορτίου” εξετάζεται εαν η συμπεριφορά ενός καταναλωτή θεωρείται προβέψιμη ή όχι. Σε αυτόν τον όρο αναφέρεται και η εντροπία της οποίας η μέτρηση μπορεί να έρθει εφαρμόζοντας k-means αλγόριθμο συσταδοποίησης στα καθημερινές τιμές φορτιού. Έτσι εαν προκύψει ότι το προφίλ φορτίου ανήκει σε πολλές διαφορετικές συστάδες, τότε το φορτίο χαρακτηρίζεται από υψηλή εντροπία, συνεπώς και είναι “απρόβλεπτο”. 

Contribution

- targeted analysis
- clustering w/ custom DTW
- peak perforance score (PPS) eval metric for clustering
- clustering algos (k-means, k-medoids, hierarchical) & comparison
- load profile analysis and description
- formalization for discretizing the real values of cluster entropy and scale awareness
- DR schemes
- low comlexity proposed methodology

Η εργασία προσφέρει μια απλή για χρήση μεθοδολογία βασισμένη στην συσταδοποίηση, με σκοπό την χρήση της από εταιρείες συλλογής για τον σχεδιασμό πολιτικών Α/Ζ σε ευέλικτες ενεργειακές κοινότητας. Σύμφωνα με την μεθοδολογία ακολουθείται η αναγκαία διαδικασία ανάλυσης και προ-επεξεργασίας δεδομένων πριν την εφαρμογή αλγορίθμων συσταδοποίησης. Η εφαρμογή των οποίων έγινε στις ημερήσιες καμπύλες φορτίου όλου του συνόλου δεδομένων, επιτρέποντας έτσι την αναγνώριση προτύπων φορτίων σε διαφορετικούς μήνες, εποχές και έτη. 

Οι αλγόριθμοι που εφαρμόστηκαν είναι : 

- k-means, ως ο τυπικός και χαμηλής πολυπλοκότητας αλγόριθμος που χρησιμοποιείται
- k-medoids
- Ιεραρχικός

Το κύριο πλεονέκτημα των k-medoids έναντι του αλγορίθμου k-means είναι η ανθεκτικότητά του σε ακραίες τιμές (outliers). Ωστόσο, είναι υπολογιστικά πολυπλοκότερος από τον αλγόριθμο k-means, ιδιαίτερα για μεγάλα σύνολα δεδομένων. Επίσης σημαντική κρίνεται η χρήση μια έκδοσης της απόστασης DTW η οποία υπολογίζει την απόσταση μεταξύ σημείων χρονοσειρών που απέχουν μεταξύ τους όχι περισσότερο από ένα χρονικό βήμα. Με την απόσταση αυτή ικανοποιείται καλύτερα ο διαχωρισμός σε cluster με βάση τα δύο κυριότερα χαρακτηριστικά των χρονοσειρών (1) οι χρονικές στιγμές των ακμών στην καμπύλη φορτίου (2) το σχήμα των καμπυλών φορτίου.

Οι μετρικές αξιολόγησης που χρησιμοποιήθηκαν ειναι οι :

- Silhouette score DTW (similar to silhouette score but calculated on DTW rather than euclidean distance)
- Davies–Bouldin validity index
- Peak match score
- Peak performance score (novel metric proposed within this
study)

Peak Performance Score (PPS):
Η εισαγωγή της συγκεκριμένης μετρικής ήρθε να καλύψει την αδυναμία του PMS στην αξιολόγηση της συσταδοποίησης βασισμένη σε σημεία ακμών ηλεκτρικού φορτίου, η οποία οφείλεται στο ότι αυτή η μετρική αξιολογεί αρνητικά μια λάθος ανίχνευση μόνο όταν το δείγμα δεν περιέχει καθόλου ακμές. Με την PPS αξιολογείται και η λάθος και η σωστή ανίχνευση ακμών του δείγματος μέσα στο cluster που εισάγεται. Έχει πεδίο τιμών [0,1].

Στα συγκριτικά αποτελέσματα, το μοντέλο με την καλύτερη επίδοση έχει τα παρακάτω χαρακτηριστικά. 

| Algorithm | Dist. Measure | # of clusters | PPS | Silhouette DTW |
| --- | --- | --- | --- | --- |
| k-means | DTW | 14 | 0.689 | 0.256 |

Σημαντική συνεισφορά στην αξιολόγηση και οπτική κατανόηση παρέχουν τα παρακάτω διαγράμματα.