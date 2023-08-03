%Read CSV file
filename = 'C:\Users\ernie\OneDrive\Desktop\Machine learning\test.csv'; %replace dataset file path with your own path location.
data = readtable(filename, 'Delimiter', ',', 'VariableNamingRule', 'preserve'); %preserve to prevent matlabs formatting/modifying column names.

% Read positive and negative words
wordListFile = 'C:\Users\ernie\OneDrive\Desktop\Machine learning\Positive and Negative Word List.xlsx'; %replace my word list file path with your own path location.
wordList = readtable(wordListFile, 'VariableNamingRule', 'preserve'); %preserve variables
positiveWords = string(wordList{:,3}); % Positive words are in the third column
negativeWords = string(wordList{:,2}); % Negative words are in the second column

% Extract necessary columns
data = data(:,{'textID','text','sentiment'}); % can include more for more data but im after the specific columns related to the text sentiment.

% Encode sentiment as numerical values
data.sentiment = categorical(data.sentiment);
data.sentiment = grp2idx(data.sentiment);

% Extract and clean documents
documents = data.text;
documents = regexprep(documents, 'https?://\S+', ''); % Remove URLs from the doucment as they provide no useful infomation for sentiment analyis for our task apart for maybe the words in the url.
documents = eraseTags(documents); % Remove HTML tags
documents = lower(documents); % Convert to lowercase
disp(documents(1:5)); % Check the documents data at five index positions.
documentsCellArray = documents;
documentsCellArray = addNGrams(documentsCellArray, 3);  % apply ngrams to the document
documentsCellArray = tokenizedDocument(documentsCellArray); % tokenize the document.


% Data Augmentation with synonym replacement
documentsCellArray = augmentData(documentsCellArray, positiveWords, negativeWords);

% Remove stop words and punctuation, and normalize words
documentsCellArray = erasePunctuation(documentsCellArray);
documentsCellArray = removeStopWords(documentsCellArray);
documentsCellArray = normalizeWords(documentsCellArray);

% Create The TF-IDF matrix
documentTfidf = tfidf(documentsCellArray);

% Normalize Features
documentTfidf = normalize(documentTfidf, 'center', 'mean', 'scale', 'std');

% Tokenize the documents
tokens = tokenizedDocument(documentsCellArray);

% Create a vocabulary set
vocab = unique(tokens.Vocabulary);

% Find which columns in TF-IDF matrix correspond to positive and negative words
negativeMatches = false(size(vocab));
positiveMatches = false(size(vocab));
for i = 1:length(vocab)
    negativeMatches(i) = any(contains(negativeWords, vocab{i}, 'IgnoreCase', true));
    positiveMatches(i) = any(contains(positiveWords, vocab{i}, 'IgnoreCase', true));
end

% Count the number of positive and negative words in each document
negativeCounts = sum(documentTfidf(:, negativeMatches), 2);
positiveCounts = sum(documentTfidf(:, positiveMatches), 2);

% Train Word2Vec model
emb = trainWordEmbedding(documentsCellArray); %emb now represents the word2vector documentcell arrays with ngrams and word frequencys which is 
                                              % used in sequences with the additional original documents variable.

positiveCounts = positiveCounts(:)';  % transpose positiveCounts to a row vector
negativeCounts = negativeCounts(:)';  % transpose negativeCounts to a row vector

% Convert tokenizedDocument to sequences of word vectors
cellDocuments = tokenizedDocument2CellArray(documents); %uses documents from line 22 converting it and returning as i have it currently for original document as a cell array which it already is, using both the original document which is less manipulated and with the additional 
sequences = cellfun(@(c) word2index(emb, c), cellDocuments, 'UniformOutput', false); %documentscellarray which contains additional features and is more heavily modified compared to original document variable in line 22 such as using (ngrams,td-idf) in documentscellarray var
                                                                                      % for model to evaluate within the sequences variable between original less modified version and the more modified version.

% Pad the sequences to have the same length
maxSeqLength = max(cellfun(@numel, sequences));
sequences = cellfun(@(c) padSequence(c, maxSeqLength, emb), sequences, 'UniformOutput', false);

% Convert sequences from indices to embeddings
sequences = cellfun(@(seq) index2embedding(emb, seq), sequences, 'UniformOutput', false);

% Add positive and negative counts as additional dimensions to each word vector
sequences = arrayfun(@(pos, neg, idx) cellfun(@(seq) appendCounts(seq, pos, neg, documentTfidf(idx, :)), sequences(idx), 'UniformOutput', false), ...
    positiveCounts, negativeCounts, 1:length(sequences), 'UniformOutput', false);



sequences = [sequences{:}];  % flatten the cell array to remove dimensions.

disp(size(sequences));
disp(size(data.sentiment));

% Split the data into training, validation and test sets
cvp = cvpartition(length(sequences), 'HoldOut', 0.2); 
idxTemp = cvp.training; % intermediate index
idxTest = cvp.test;
disp(size(idxTemp));
disp(size(idxTest));
% Further split the training data into actual training and validation sets
cvp2 = cvpartition(length(sequences(idxTemp)), 'HoldOut', 0.2); 64% for training. 64% training, 16% validation,20% testing.
idxTrain = idxTemp(cvp2.training);
idxValidation = idxTemp(cvp2.test);
disp(size(idxTrain));
disp(size(idxValidation));
% Training, validation, and test data
XTrain = sequences(idxTrain);
YTrain = data.sentiment(idxTrain);

XValidation = sequences(idxValidation);
YValidation = data.sentiment(idxValidation);

XTest = sequences(idxTest);
YTest = data.sentiment(idxTest);

disp(size(XTrain));
disp(size(YTrain));
disp(size(XValidation));
disp(size(YValidation));
disp(size(XTest));
disp(size(YTest));

% Save the preprocessed data, 7.3v for file size limits. 
save('preprocessedData.mat', 'XTrain', 'YTrain', 'XValidation', 'YValidation', 'XTest', 'YTest', 'emb', '-v7.3');

% Helper function for finding word index in embedding
function indices = word2index(emb, words)
    indices = cell(size(words));
    for i = 1:numel(words)
        idx = find(ismember(emb.Vocabulary, words(i))); 
        if ~isempty(idx)
            indices{i} = idx;
        end
    end
    indices(cellfun('isempty', indices)) = [];  % remove empty cells
    indices = cell2mat(indices);  % convert cell array to numeric array matrice
    indices = indices(indices <= numel(emb.Vocabulary));  % Ensure indices do not exceed vocabulary size
end

% Pad sequences to the same length
function c = padSequence(c, maxSeqLength, emb)
    padIndex = numel(emb.Vocabulary);  % index for padding (use the last vocabulary index)
    if numel(c) < maxSeqLength
        padding = repmat(padIndex, 1, maxSeqLength - numel(c));
        c = [c(:); padding(:)];  % make sure both arrays are column vectors before concatenating
    elseif numel(c) > maxSeqLength
        c = c(1:maxSeqLength);
    end
    c = c(:)';  % make sure the output is a row vector
end

function cellArray = tokenizedDocument2CellArray(docArray)
    cellArray = docArray; %input as output due to correct format, can be used if we wish to perform more tokenization formatting steps.
end                        

% Helper function to convert indices to word embeddings
function embeddings = index2embedding(emb, indices)
    embeddings = zeros(length(indices), emb.Dimension);
    for i = 1:length(indices)
        if indices(i) <= numel(emb.Vocabulary)  % check if index is within the range
            embeddings(i, :) = emb.word2vec(emb.Vocabulary{indices(i)});
        else
            disp(['Index ', num2str(indices(i)), ' exceeds the vocabulary size. Skipping...']);
        end
    end
end

% Helper function to append counts to each word vector
function c = appendCounts(c, pos, neg, tfidfFeatures)
    pos_vec = repmat(pos, size(c, 1), 1);
    neg_vec = repmat(neg, size(c, 1), 1);
    tfidf_vec = repmat(tfidfFeatures, size(c, 1), 1); % Appending the tfidf as vector matrices as a additional feature for the model to learn from.
    c = [c, pos_vec, neg_vec, tfidf_vec]; 
end


% Data Augmentation Function
function augmentedDocs = augmentData(docs, negativeWords, positiveWords)
    % Placeholder for augmented data
    augmentedDocs = tokenizedDocument.empty;
    
    for i = 1:numel(docs)
        docTokens = string(docs(i).tokenDetails.Token);
        for j = 1:numel(docTokens)
            % Randomly choose whether to replace the word with a synonym
            if rand() < 0.2 % 20% chance to replace the word to improve real world model results.
                synonyms = getSynonyms(docTokens(j), negativeWords, positiveWords);
                if ~isempty(synonyms)
                    % Replace the word with a randomly chosen synonym
                    docTokens(j) = synonyms(randi(length(synonyms)));
                end
            end 
        end  
        augmentedDocs = [augmentedDocs; tokenizedDocument(strjoin(docTokens))];

    end %replacing positive words with other positive words in the sentence to get more robust accuracy results, same for negative words.
end % for example : i [like] dogs will be replaced with, i [love] dogs and vice versa for negative words.

function synonyms = getSynonyms(word, positiveWords, negativeWords)
    % Utilize positive and negative word lists to build synonyms
    synonyms = [];
    if any(positiveWords == word)
        synonyms = positiveWords(positiveWords ~= word); %  ( ~ ) same as ( != ).
    elseif any(negativeWords == word)
        synonyms = negativeWords(negativeWords ~= word);
    end
end

function documentsWithNGrams = addNGrams(documents, ngramLength)
    documentsWithNGrams = cell(size(documents));
    for i = 1:numel(documents)
        tokens = strsplit(documents{i});
        ngrams = strings(0, 1);
        if numel(tokens) >= ngramLength
            for j = 1:numel(tokens) - ngramLength + 1
                ngram = strjoin(tokens(j:j+ngramLength-1), ' ');
                ngrams = [ngrams; ngram];
            end
        end
        combinedTokens = [tokens, ngrams']; % Use horizontal concatenation
        documentsWithNGrams{i} = strjoin(combinedTokens);
    end
end

function tfidfMatrix = tfidf(docs)
    % Tokenize the documents
    tokens = tokenizedDocument(docs);
    
    % Compute the term frequencies (TF)
    N = numel(tokens);
    vocab = tokens.Vocabulary;
    tf = zeros(N, numel(vocab));
    for i = 1:N
        tokenDocument = tokens(i).tokenDetails.Token;
        totalTokens = numel(tokenDocument);
        for j = 1:numel(vocab)
            tokenCounts = sum(strcmp(tokenDocument, vocab(j)));
            tf(i, j) = tokenCounts / totalTokens;
        end
    end
    
    % Compute the inverse document frequency (IDF)
    df = sum(tf > 0, 1);
    idf = log(N ./ df);
    
    % Compute TF-IDF
    tfidfMatrix = tf .* idf; %term frequency used to obtain word freq so we can determine word importance in each sentence.
end
