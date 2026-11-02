import numpy as np
import torch
import torch.nn as nn

from student_lab import ActivationFunctions, Optimizers, Dropout, BatchNorm, setup_transfer_learning

def test_activation_functions():
    print("--- Testing Activation Functions ---")
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    
    # Test Sigmoid
    out_sig = ActivationFunctions.sigmoid(x)
    expected_sig = np.array([0.26894142, 0.5, 0.73105858, 0.88079708])
    if out_sig is not None:
        np.testing.assert_almost_equal(out_sig, expected_sig, decimal=5, err_msg="Sigmoid forward failed!")
        print("Sigmoid Forward: ✅")
    else:
        print("Sigmoid Forward: (Not Implemented)")
    
    # Test Sigmoid Backward
    dout = np.ones_like(x)
    dx_sig = ActivationFunctions.sigmoid_backward(dout, x)
    expected_dx_sig = np.array([0.19661193, 0.25, 0.19661193, 0.10499359])
    if dx_sig is not None:
        np.testing.assert_almost_equal(dx_sig, expected_dx_sig, decimal=5, err_msg="Sigmoid backward failed!")
        print("Sigmoid Backward: ✅")
    else:
        print("Sigmoid Backward: (Not Implemented)")
    
    # Test ReLU
    out_relu = ActivationFunctions.relu(x)
    expected_relu = np.array([0.0, 0.0, 1.0, 2.0])
    if out_relu is not None:
        np.testing.assert_almost_equal(out_relu, expected_relu, decimal=5, err_msg="ReLU forward failed!")
        print("ReLU Forward: ✅")
    else:
        print("ReLU Forward: (Not Implemented)")
    
    print("Activation Functions section finished.\n")


def test_optimizers():
    print("--- Testing Optimizers ---")
    w = np.array([1.0, 2.0])
    dw = np.array([0.1, -0.2])
    
    # Test SGD + Momentum
    v = np.array([0.01, -0.01])
    try:
        w_next, v_next = Optimizers.sgd_momentum(w.copy(), dw, v.copy(), lr=0.1, rho=0.9)
        expected_v = np.array([0.109, -0.209])
        expected_w = np.array([0.9891, 2.0209])
        np.testing.assert_almost_equal(v_next, expected_v, decimal=4, err_msg="SGD+Momentum velocity update failed!")
        np.testing.assert_almost_equal(w_next, expected_w, decimal=4, err_msg="SGD+Momentum weight update failed!")
        print("SGD + Momentum: ✅")
    except Exception as e:
        print(f"SGD + Momentum: ❌ (Error or Not Implemented)")
    
    # Test Adam
    m = np.array([0.0, 0.0])
    v_adam = np.array([0.0, 0.0])
    try:
        w_adam_next, m_next, v_adam_next = Optimizers.adam(w.copy(), dw, m, v_adam, t=1, lr=0.1)
        # Apply bias correction check
        np.testing.assert_almost_equal(m_next, np.array([0.01, -0.02]), decimal=4, err_msg="Adam 'm' update failed!")
        np.testing.assert_almost_equal(v_adam_next, np.array([1e-5, 4e-5]), decimal=6, err_msg="Adam 'v' update failed!")
        assert w_adam_next[0] < 1.0, "Adam should decrease w[0]"
        assert w_adam_next[1] > 2.0, "Adam should increase w[1]"
        print("Adam Optimizer: ✅")
    except Exception as e:
        print(f"Adam Optimizer: ❌ (Error or Not Implemented)")
    
    print("Optimizers section finished.\n")


def test_dropout():
    print("--- Testing Inverted Dropout ---")
    x = np.ones((1000, 1000)) # Large matrix to utilize law of large numbers
    p = 0.5
    
    # Test Train mode
    try:
        out_train, mask = Dropout.dropout_forward(x, p=p, mode='train')
        if out_train is not None:
            # 1. Verify output elements are either 0 or x/p (i.e., 2.0)
            unique_vals = np.unique(out_train)
            assert np.allclose(unique_vals, [0.0, 2.0]), "Inverted Dropout train values should be 0 or 1/p!"
            
            # 2. Verify retention rate is roughly p
            retain_rate = np.mean(mask > 0) 
            assert 0.45 < retain_rate < 0.55, f"Expected retain rate around {p}, got {retain_rate}"
            print("Dropout (Train): ✅")
            
            # Test Test mode
            out_test, _ = Dropout.dropout_forward(x, p=p, mode='test')
            assert np.array_equal(out_test, x), "Inverted Dropout test mode should be identity!"
            print("Dropout (Test): ✅")
        else:
            print("Dropout: (Not Implemented)")
    except Exception as e:
        print(f"Dropout: ❌ (Error or Not Implemented)")
    
    print("Dropout section finished.\n")


def test_batchnorm():
    print("--- Testing Batch Normalization ---")
    np.random.seed(42)
    x = np.random.randn(100, 5) * 2.0 + 5.0 # mean ~ 5, std ~ 2
    
    gamma = np.ones(5)
    beta = np.zeros(5)
    running_mean = np.zeros(5)
    running_var = np.ones(5)
    
    # Test Train mode
    try:
        out_train, rm_next, rv_next = BatchNorm.batchnorm_forward(x, gamma, beta, running_mean, running_var, mode='train')
        if out_train is not None:
            # Train mode: batch mean should be ~0, variance ~1
            np.testing.assert_almost_equal(np.mean(out_train, axis=0), np.zeros(5), decimal=4, err_msg="BN train mean is not 0!")
            np.testing.assert_almost_equal(np.var(out_train, axis=0), np.ones(5), decimal=4, err_msg="BN train variance is not 1!")
            
            # Verify running statistics update
            assert np.any(rm_next != 0), "Running mean was not updated!"
            assert np.any(rv_next != 1), "Running variance was not updated!"
            print("BN (Train): ✅")
            
            # Test Test mode
            out_test, _, _ = BatchNorm.batchnorm_forward(x, gamma, beta, rm_next, rv_next, mode='test')
            # Test mode should not use current batch statistics
            assert not np.allclose(np.mean(out_test, axis=0), np.zeros(5)), "BN test should not use batch statistics!"
            print("BN (Test): ✅")
        else:
            print("BN: (Not Implemented)")
    except Exception as e:
        print(f"BN: ❌ (Error or Not Implemented)")
    
    print("Batch Normalization section finished.\n")


def test_transfer_learning():
    print("--- Testing Transfer Learning ---")
    num_classes = 10
    try:
        model = setup_transfer_learning(num_classes=num_classes)
        if model is not None:
            # 1. Verify frozen layers
            frozen_layers_count = 0
            for name, param in model.named_parameters():
                if "fc" not in name: 
                    assert not param.requires_grad, f"Parameter {name} should be frozen!"
                    frozen_layers_count += 1
                else:
                    assert param.requires_grad, f"Final layer parameter {name} should be trainable!"
            
            assert frozen_layers_count > 0, "No layers were frozen!"
            
            # 2. Verify output dimension
            assert isinstance(model.fc, nn.Linear), "Final layer must be an nn.Linear layer!"
            assert model.fc.out_features == num_classes, f"Expected output features to be {num_classes}!"
            print("Transfer Learning: ✅")
        else:
            print("Transfer Learning: (Not Implemented)")
    except Exception as e:
        print(f"Transfer Learning: ❌ (Error or Not Implemented)")
    
    print("Transfer Learning section finished.\n")


def run_all_tests():
    print("🚀 Starting Lab Week 07 Verification Scripts...")
    test_activation_functions()
    test_optimizers()
    test_dropout()
    test_batchnorm()
    test_transfer_learning()
    print("🎉 Verification Finished!")

if __name__ == "__main__":
    run_all_tests()
