using MLDatasets
using MultivariateStats
using Statistics

function mnist01_pca_data(n_components::Int, n_train_per_class::Int, n_test_per_class::Int)

    train_imgs, train_labels = MNIST.traindata()
    test_imgs,  test_labels  = MNIST.testdata()

    train_labels = collect(Int.(train_labels))
    test_labels  = collect(Int.(test_labels))

    function select_01(imgs, labels, n_per_class)
        idx0 = findall(==(0), labels)
        idx1 = findall(==(1), labels)

        idx0 = idx0[1:min(n_per_class, length(idx0))]
        idx1 = idx1[1:min(n_per_class, length(idx1))]

        idx = vcat(idx0, idx1)
        shuffle!(idx)

        N = length(idx)
        X = zeros(Float64, 28*28, N)   # d × N (d=784)
        y = zeros(Int, N)

        for (k, i) in enumerate(idx)
            img = Float64.(imgs[:, :, i]) ./ 255.0  # [0,1] normalize
            X[:, k] .= reshape(img, :, 1)          # kolon = 1 örnek
            y[k] = labels[i]
        end

        return X, y
    end

    Xtr, ytr = select_01(train_imgs, train_labels, n_train_per_class)
    Xte, yte = select_01(test_imgs,  test_labels,  n_test_per_class)

    # ---- PCA: gözlemler sütunlarda, feature'lar satırlarda ----
    # Xtr: 784 × N_train
    pca_model = fit(PCA, Xtr; maxoutdim=n_components)

    # transform aynı yönü korur:  -> n_components × N
    Ztr = MultivariateStats.transform(pca_model, Xtr)  # n_components × N_train
    Zte = MultivariateStats.transform(pca_model, Xte)  # n_components × N_test

    # İsteğe bağlı: her feature'ı [0,1] aralığına ölçekle
    function minmax_scale!(X)
        for i in 1:size(X, 1)  # satırlar: feature
            xi = X[i, :]
            xmin = minimum(xi)
            xmax = maximum(xi)
            if xmax > xmin
                X[i, :] .= (xi .- xmin) ./ (xmax - xmin)
            else
                X[i, :] .= 0.5
            end
        end
        return X
    end

    minmax_scale!(Ztr)
    minmax_scale!(Zte)

    # Artık Ztr ve Zte zaten (n_components × N)
    X_train_pca = Ztr
    X_test_pca  = Zte

    return X_train_pca, ytr, X_test_pca, yte
end
