function [value,isterminal,direction] = myProfMotionEvent(t,z)
    value = z(2);     % Detect height = 0
    isterminal = 1;   % Stop the integration
    direction = -1;   % Negative direction only
end