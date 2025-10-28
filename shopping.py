import csv
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Load data from CSV file
    evidence, labels = load_data("shopping.csv")

    # Split data into training and testing sets
    train_size = int(0.8 * len(evidence))
    train_evidence = evidence[:train_size]
    train_labels = labels[:train_size]
    test_evidence = evidence[train_size:]
    test_labels = labels[train_size:]

    # Train model
    model = train_model(train_evidence, train_labels)

    # Make predictions on test set
    predictions = model.predict(test_evidence)

    # Evaluate model
    sensitivity, specificity = evaluate(test_labels, predictions)

    # Print results
    print(f"Correct: {sum(1 for true, pred in zip(test_labels, predictions) if true == pred)}")
    print(f"Incorrect: {sum(1 for true, pred in zip(test_labels, predictions) if true != pred)}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file and convert it into a list of evidence lists and a list of labels.
    """
    evidence = []
    labels = []
    
    month_map = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Process evidence
            evidence_row = [
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                month_map[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ]
            evidence.append(evidence_row)
            
            # Process label
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)
    
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a trained k-nearest neighbor classifier.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    true_positives = 0
    actual_positives = 0
    true_negatives = 0
    actual_negatives = 0
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            actual_positives += 1
            if predicted == 1:
                true_positives += 1
        else:
            actual_negatives += 1
            if predicted == 0:
                true_negatives += 1
    
    sensitivity = true_positives / actual_positives
    specificity = true_negatives / actual_negatives
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()