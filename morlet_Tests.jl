using LinearAlgebra
using Plots

# t = tempo
# μ = posicionamento temporal da janela gaussiana
# ω = frequencia das oscilaçoes
# N = numero de oscilacoes na janela

function morlet(t, μ, ω, N)
    # Tamanho da janela gaussiana
    a = log(1/0.05)*(ω/(π*N))^2
    # Parametro de y-shift
    b = exp(-ω^2/(4*a))
    # Constante de normalizacao
    c = sqrt(1/(sqrt(π/2/a)*(1/2 + b^2 + exp(-ω^2/a) - 2*b*exp(-ω^2/8/a))))

    real = c .* exp.(-a .* (t .- μ).^2) .* cos.(ω .* (t .- μ))
    imag = c .* exp.(-a .* (t .- μ).^2) .* sin.(ω .* (t .- μ))
    return real, imag
end

function ricker(t, μ, δ)
    c = 2/sqrt(3*δ)*π^(1/4)
    w = 1 .- (t ./ δ).^2
    f = c .* w .* exp.(-(t.^2)./(2*δ^2))
    return f
end


N = 5
ω = 10
μ = 0.0
t = range(-2*π*(N+1)/ω, 2*π*(N+1)/ω, length=1000)
a, b = morlet(t, μ, ω, N)
plot(t, a, color = :blue)
plot!(t, b, color = :red)
