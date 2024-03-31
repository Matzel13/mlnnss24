%% Wurfbahn eines Balles
load("BallwurfDaten.mat");
scatter(xmess, ymess, 'filled', 'Displayname', 'Messpunkte');
axis([0, 35, 0, 12]);
xlabel("Weite"), ylabel("Höhe"); 
title("Flugbahn eines Balls");
legend("Location", "NW");




%% die vollstaendige Flugbahn
hold on;
plot(z1, z2, 'Displayname', 'Flugbahn', 'LineWidth', 2);
hold off; 
