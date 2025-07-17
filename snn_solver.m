function [t, X] = snn_solver(A, b, C, d, t_end, x0, k0, k1)
    % min x^TAx/2+b^Tx, st.Cx+d<=0

    t_store = cell(1);
    x_store = cell(1);

    t_store{1} = 0;
    x_store{1} = x0.';

    tspan = [0, t_end];

    function [value, isterminal, direction] = myEvents(~, x)
        y = C*x + d;
        value = all(y <= 0);
        isterminal = 1;
        direction = 0;
    end

    function dotx = myode(~, x)
        fgrad = A*x + b;
        dotx = -k0*fgrad;
    end

    options = odeset('Events', @myEvents, 'MaxStep', 0.1);
    idx = 2;
    while(1)        
        if tspan(1)>=tspan(2)
            break
        end

        % Spiking neuron model
        while(1)
            y = C*x0 + d;
            if any(y>0)
                x0 = x0 - k1*C'*(y>0);
            else
                break
            end
        end
        [t_, X_] = ode45(@myode, tspan, x0, options);
        tspan(1) = t_(end);
        x0 = X_(end, :)';

        %Data collection
        t_store{idx} = t_;
        x_store{idx} = X_;
        idx = idx + 1;
    end

    t = cat(1, t_store{:});
    X = cat(1, x_store{:});
end
