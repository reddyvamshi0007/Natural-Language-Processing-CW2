import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """
    Set seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_smaller_dataset(input_file, output_file, max_size_mb=2, chunk_size=10000):
    """
    Create a smaller version of the dataset by sampling rows.
    """
    total_size = 0
    samples = []
    
    # Open the large file and read in chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read header
        header = f.readline().strip()
        samples.append(header)
        
        while True:
            # Read a chunk of lines
            chunk = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                chunk.append(line)
            
            if not chunk:
                break
                
            # Randomly sample from this chunk
            sample_size = min(100, len(chunk))  # Adjust this number as needed
            selected = np.random.choice(chunk, size=sample_size, replace=False)
            
            # Add selected lines to samples
            samples.extend(selected)
            
            # Check current file size
            current_size = sum(len(s.encode('utf-8')) for s in samples) / (1024 * 1024)  # Size in MB
            if current_size >= max_size_mb:
                break
    
    # Write the sampled data to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(samples)
    
    return output_file

def clean_text(text):
    """Clean and preprocess text"""
    # Remove special characters and extra whitespace
    text = ' '.join(text.split())
    return text.lower()

def perform_eda(data):
    """
    Perform exploratory data analysis on the dataset
    """
    plt.figure(figsize=(15, 5))
    
    # Text length distribution with KDE
    plt.subplot(1, 3, 1)
    text_lengths = [len(str(text).split()) for text in data['text']]
    sns.histplot(text_lengths, kde=True)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    # Most common words (excluding stopwords)
    plt.subplot(1, 3, 2)
    # Get word frequencies from actual text content
    word_freq = Counter()
    for text in data['text']:
        words = clean_text(str(text)).split()
        # Filter out very short words and numbers
        words = [word for word in words if len(word) > 2 and not word.isdigit()]
        word_freq.update(words)
    
    # Plot top 10 most common words
    common_words = dict(word_freq.most_common(10))
    plt.bar(common_words.keys(), common_words.values())
    plt.title('Top 10 Most Common Words')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Frequency')
    
    # Label distribution
    plt.subplot(1, 3, 3)
    label_counts = Counter(data['label'])
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', bbox_inches='tight')
    plt.close()
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Total number of samples: {len(data['text'])}")
    print(f"Average text length: {np.mean(text_lengths):.2f} words")
    print(f"Median text length: {np.median(text_lengths):.2f} words")
    print(f"Max text length: {max(text_lengths)} words")
    print(f"Min text length: {min(text_lengths)} words")
    print(f"Number of unique labels: {len(set(data['label']))}")
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count}")
    
    print("\nTop 10 most common words:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")
    
    print("\nSample entries:")
    for i in range(min(3, len(data['text']))):
        print(f"\nExample {i+1}:")
        print(f"Text: {data['text'][i][:200]}...")
        print(f"Label: {data['label'][i]}")
        if 'year' in data:
            print(f"Year: {data['year'][i]}")

class TextPreprocessor:
    def __init__(self, max_vocab_size=5000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {}
        self.idx2word = {}
        
    def build_vocabulary(self, texts):
        word_counts = Counter()
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
                word_counts.update(words)
        
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(self.max_vocab_size - 2)]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def text_to_sequence(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        words = text.lower().split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if len(sequence) < self.max_seq_length:
            sequence += [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))
        else:
            sequence = sequence[:self.max_seq_length]
            
        return torch.tensor(sequence)

class RCTDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.preprocessor.text_to_sequence(text)
        return sequence, torch.tensor(label)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]
        output = self.fc(hidden)
        return output

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output = self.transformer(embedded)
        output = output.mean(dim=1)  # Global average pooling
        return self.fc(output)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in train_loader:
        sequences, labels = batch
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(sequences)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    return total_loss / len(train_loader), accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            output = model(sequences)
            loss = criterion(output, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    return total_loss / len(test_loader), accuracy

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_history.png')
    plt.close()

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, model_name):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    plot_training_history(history, model_name)
    return history

def classify_statement(model, statement, preprocessor, device):
    """
    Classify a single statement using the trained model
    """
    model.eval()
    # Preprocess the statement
    sequence = preprocessor.text_to_sequence(statement)
    # Add batch dimension
    sequence = sequence.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # File paths
    input_file = 'rct_data.txt'
    small_data_file = 'rct_data_small.txt'
    
    # Create smaller dataset
    small_data_file = create_smaller_dataset(input_file, small_data_file, max_size_mb=2)
    
    # Read the smaller dataset
    data = {'text': [], 'label': []}
    with open(small_data_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:  # We need ID, label, year, title, and text
                try:
                    text = parts[4]  # Text is the last column
                    label = int(parts[1])  # Label is the second column
                    data['text'].append(text)
                    data['label'].append(label)
                except (ValueError, IndexError):
                    continue  # Skip invalid lines
    
    if len(data['text']) == 0:
        raise ValueError("No valid data was read from the file. Please check the file format.")
    
    print("\nSample entries:")
    for i in range(min(3, len(data['text']))):
        print(f"\nExample {i+1}:")
        print(f"Text: {data['text'][i][:200]}...")
        print(f"Label: {data['label'][i]}")
    
    # Perform EDA
    perform_eda(data)
    
    # Use the data for training
    texts = data['text']
    labels = data['label']
    
    # Split data with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize preprocessor and build vocabulary
    preprocessor = TextPreprocessor()
    preprocessor.build_vocabulary(X_train)
    
    # Create datasets and dataloaders
    train_dataset = RCTDataset(X_train, y_train, preprocessor)
    test_dataset = RCTDataset(X_test, y_test, preprocessor)
    
    # Use a fixed seed for DataLoader workers
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                            worker_init_fn=lambda x: np.random.seed(42), 
                            generator=g)
    test_loader = DataLoader(test_dataset, batch_size=32,
                           worker_init_fn=lambda x: np.random.seed(42),
                           generator=g)
    
    # Initialize models and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LSTM Model with fixed initialization
    torch.manual_seed(42)  # Reset seed before model initialization
    lstm_model = LSTMClassifier(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=100,
        hidden_dim=256,
        output_dim=len(set(labels)),
        n_layers=2,
        dropout=0.2
    ).to(device)
    
    # Transformer Model with fixed initialization
    torch.manual_seed(42)  # Reset seed before model initialization
    transformer_model = TransformerClassifier(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=100,
        nhead=4,
        num_layers=2,
        output_dim=len(set(labels)),
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train LSTM
    print("\nTraining LSTM Model...")
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_history = train_model(
        lstm_model, train_loader, test_loader,
        criterion, lstm_optimizer, device, num_epochs=10,
        model_name='LSTM'
    )
    
    # Train Transformer
    print("\nTraining Transformer Model...")
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.0005)
    transformer_history = train_model(
        transformer_model, train_loader, test_loader,
        criterion, transformer_optimizer, device, num_epochs=10,
        model_name='Transformer'
    )
    
    # Compare final results
    print("\nFinal Results:")
    print("LSTM Model:")
    lstm_val_acc = lstm_history['val_acc'][-1]
    print(f"Train Accuracy: {lstm_history['train_acc'][-1]:.4f}")
    print(f"Validation Accuracy: {lstm_val_acc:.4f}")
    print("\nTransformer Model:")
    transformer_val_acc = transformer_history['val_acc'][-1]
    print(f"Train Accuracy: {transformer_history['train_acc'][-1]:.4f}")
    print(f"Validation Accuracy: {transformer_val_acc:.4f}")
    
    # Determine the best model
    best_model = lstm_model if lstm_val_acc > transformer_val_acc else transformer_model
    print("\nBest Model:")
    if lstm_val_acc > transformer_val_acc:
        print("LSTM Model performed better with validation accuracy: {:.4f}".format(lstm_val_acc))
        print("(Outperformed Transformer by {:.4f})".format(lstm_val_acc - transformer_val_acc))
    elif transformer_val_acc > lstm_val_acc:
        print("Transformer Model performed better with validation accuracy: {:.4f}".format(transformer_val_acc))
        print("(Outperformed LSTM by {:.4f})".format(transformer_val_acc - lstm_val_acc))
    else:
        print("Both models performed equally with validation accuracy: {:.4f}".format(lstm_val_acc))
    
    # Example statements for classification
    print("\nClassifying example statements:")
    example_statements = [
        "A randomized controlled trial showed significant improvement in patient outcomes.",
        "This paper reviews the current literature on treatment methods.",
        "The study examined 100 patients over a period of 6 months.",
        # Add more example statements as needed
    ]
    
    print("\nClassification Results:")
    for statement in example_statements:
        prediction, confidence = classify_statement(best_model, statement, preprocessor, device)
        print(f"\nStatement: {statement}")
        print(f"Predicted class: {prediction}")
        print(f"Confidence: {confidence:.4f}")
    
    # Interactive classification
    print("\nEnter statements to classify (type 'quit' to exit):")
    while True:
        statement = input("\nEnter a statement: ")
        if statement.lower() == 'quit':
            break
        
        prediction, confidence = classify_statement(best_model, statement, preprocessor, device)
        print(f"Predicted class: {prediction}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main() 