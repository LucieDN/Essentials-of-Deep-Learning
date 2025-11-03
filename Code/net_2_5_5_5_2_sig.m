function [X,Y,Mval,savecost] = net_2_5_5_5_2_sig(m, n, x1, x2, y, eta, Niter)
    tic    % begin of clock count 

    % Initialize weights and biases 
    rng('default') 
    W2 = 0.5*randn(5,2);
    W3 = 0.5*randn(5,5);
    W4 = 0.5*randn(5,5);
    W5 = 0.5*randn(2,5);
    b2 = 0.5*randn(5,1);
    b3 = 0.5*randn(5,1);
    b4 = 0.5*randn(5,1);
    b5 = 0.5*randn(2,1);

    % Backpropagation to train the network 
    savecost = zeros(Niter,1);
    for counter = 1:Niter
        k = randi(m+n);  % Choose one point from the sample
        x = [x1(k); x2(k)];
        % Forward pass
        a2 = activate(x,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        a5 = activate(a4,W5,b5);
        % Backward pass
        delta5 = a5.*(1-a5).*(a5-y(:,k));
        delta4 = a4.*(1-a4).*(W5'*delta5);
        delta3 = a3.*(1-a3).*(W4'*delta4);
        delta2 = a2.*(1-a2).*(W3'*delta3);
        % Gradient step
        W2 = W2 - eta*delta2*x';
        W3 = W3 - eta*delta3*a2';
        W4 = W4 - eta*delta4*a3';
        W5 = W5 - eta*delta5*a4';
        b2 = b2 - eta*delta2;
        b3 = b3 - eta*delta3;
        b4 = b4 - eta*delta4;
        b5 = b5 - eta*delta5;
        % Monitor progress
        newcost = cost(W2,W3,W4,W5,b2,b3,b4,b5);
        savecost(counter) = newcost;
    end

    disp("Total training time: " + toc)   % end of clock 

    % Compute classification zones
    N = 500;
    Dx = 1/N;
    Dy = 1/N;
    xvals = [0:Dx:1];
    yvals = [0:Dy:1];
    for k1 = 1:N+1
        xk = xvals(k1);
        for k2 = 1:N+1
            yk = yvals(k2);
            xy = [xk;yk];
            a2 = activate(xy,W2,b2);
            a3 = activate(a2,W3,b3);
            a4 = activate(a3,W4,b4);
            a5 = activate(a4,W5,b5);
            Aval(k2,k1) = a5(1); % a5 is a 2 by 1 vector to indicate class
            Bval(k2,k1) = a5(2);
        end
    end
    Mval = Aval>=Bval;
    [X,Y] = meshgrid(xvals,yvals);
    show_training_results(m, n, x1, x2, X, Y, Mval, savecost, Niter, "(2\rightarrow5\rightarrow5\rightarrow5\rightarrow2) with sigmoid training results");

    % Auxiliary functions 
    function costval = cost(W2,W3,W4,W5,b2,b3,b4,b5)
        costvec = zeros(m+n,1); 
        for i = 1:m+n
            x =[x1(i);x2(i)];
            a2 = activate(x,W2,b2);
            a3 = activate(a2,W3,b3);
            a4 = activate(a3,W4,b4);
            a5 = activate(a4,W5,b5);
            costvec(i) = norm(y(:,i) - a5,2);
        end
        costval = norm(costvec,2)^2;          
    end % of nested function

    function y = activate(x,W,b)
        % ACTIVATE Evaluates sigmoid function.
        % x is the input vector, y is the output vector
        % W contains the weights, b contains the shifts
        % The ith component of y is activate((Wx+b)_i)
        % where activate(z) = 1/(1+exp(-z))
        y = 1./(1+exp(-(W*x+b)));
    end % of nested function
end
