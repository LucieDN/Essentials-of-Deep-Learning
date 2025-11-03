function fig = show_training_results(m, n, x1, x2, X, Y, Mval, savecost, Niter, tit)
    figure('Position', [0 0 600 300]);
    clf;

    % Classification zones
    subplot(1,2,1);
    contourf(X,Y,Mval,[0.5 0.5]);
    colormap([1 1 1; 0.8 0.8 0.8]);
    hold on
    plot(x1(1:m),x2(1:m),'ro','MarkerSize',10,'LineWidth',2);
    hold on
    plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',10,'LineWidth',2);
    xticks([0 1]);
    yticks([0 1]);
    title("Classification results");

    % Cost function
    subplot(1,2,2);
    semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2);
    xlabel('Iteration Number');
    ylabel('Value of cost function');
    set(gca,'FontWeight','Bold','FontSize',10);
    title("Cost function");
    sgtitle(tit);
end