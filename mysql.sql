CREATE TABLE Tweets (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(255),
    tweet_text TEXT,
    created_at DATETIME,
    hashtag VARCHAR(255)
);

CREATE TABLE Sentiment (
    id INT PRIMARY KEY AUTO_INCREMENT,
    tweet_id INT,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(50),
    FOREIGN KEY (tweet_id) REFERENCES Tweets(id)
);

CREATE TABLE RiskClassification (
    id INT PRIMARY KEY AUTO_INCREMENT,
    tweet_id INT,
    risk_level VARCHAR(50),
    FOREIGN KEY (tweet_id) REFERENCES Tweets(id)
);
