classdef AttentionLayer < nnet.layer.Layer
    properties (Learnable)
        W1
        W2
        V
    end

    methods
        function layer = AttentionLayer(name, numHiddenUnits)
            layer.Name = name;
            layer.Description = "Attention layer";
            layer.W1 = single(randn(numHiddenUnits, numHiddenUnits) * sqrt(2/numHiddenUnits));
            layer.W2 = single(randn(numHiddenUnits, numHiddenUnits) * sqrt(2/numHiddenUnits));
            layer.V = single(randn(numHiddenUnits, 1));
        end

        function Z = predict(layer, X)
            keys = tanh(layer.W2*X);
            scores = layer.V' * keys;
            weights = softmax(scores, 'all');
            Z = sum(X .* weights, 1);
        end

function [dLdX, dLdW1, dLdW2, dLdV] = backward(layer, X, ~, dLdZ, ~)
    keys = tanh(layer.W2*X);
    scores = layer.V' * keys;
    weights = softmax(scores, 'all');

    dScores = single(dLdZ .* weights);
    dKeys = single((layer.V * dScores) .* (1 - keys.^2));
    dLdW2 = single(dKeys * X');
    dLdV = single((dScores * keys')'); % transpose the result
    dLdX = single(layer.W2' * dKeys);
    dLdW1 = single(zeros(size(layer.W1))); % Since W1 is not used, derivative with respect to it will be zero
        end
    end
end




