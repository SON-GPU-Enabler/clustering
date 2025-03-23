function [C, e, k] = clusteringCoefficients(P, isUndirected, onGPU, method)

methods = {'nnz', 'sum', ...
    'dot_product', ...
    'set_intersection_serialOverColumns', 'set_intersection_parallelOverColumns', ...
    'all_triangles_reduce_by_key', 'all_triangles_count_if', ...
    'triangles_binary_search_full_field', 'triangles_binary_search_segmented_field', 'triangles_vectorized_binary_search_full_field'};

if nargin < 4 || isnumeric(method) && (method < 1 || method > numel(methods)) || ischar(method) && ~ismember(method, methods)
    method = 'dot_product';
    if nargin < 3
        onGPU = gpuDeviceCount > 0;
    end
end

if isnumeric(method)
    method = methods{method};
end

isSym = issymmetric(P);

if nargin < 2
    isUndirected = isSym;
end

% EXTREM WICHTIG: scattering von e??!?!

if strcmpi(method, 'nnz')
    P = logical(sparse(P));
    n = size(P, 1);
    if isUndirected && ~isSym
        P = P | P';
    end
    P(1:n+1:end) = false; % eliminate diagonal
    e = zeros(n, 1);
    for i = 1:n
        ind = P(i, :)' | P(:, i);
        e(i) = nnz(P(ind, ind));
    end
    if isUndirected
        e = e / 2;
    end
    k = full(sum(double(P | P'), 2));
    if isUndirected
        C = 2 * e ./ (k .* (k - 1));
    else
        C = e ./ (k .* (k - 1));
    end
elseif strcmpi(method, 'sum')
    n = size(P, 1);
    if isUndirected && ~isSym
        P = P + P';
    end
    P = sign(P);
    P(1:n+1:end) = 0; % eliminate diagonal
    if isUndirected || isSym
        e = sum(P .* (P * P), 1)';
        if isUndirected
            e = e / 2;
        end
        k = sum(P, 1)';
        if isUndirected
            C = 2 * e ./ (k .* (k - 1));
        else
            C = e ./ (k .* (k - 1));
        end
    else
        Psym = sign(P + P');
        e = sum(Psym .* (P * Psym), 1)';
        k = sum(Psym, 1)';
        if isUndirected
            C = 2 * e ./ (k .* (k - 1));
        else
            C = e ./ (k .* (k - 1));
        end
    end
else
    [C, e, k] = clusteringCoefficients_mex(P, isUndirected, onGPU, isSym, method);
end