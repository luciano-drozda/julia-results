# Extract the reference functions from the driver (text between markers) and test them with finite differences.
src = read("adv_driver_final.jl", String)
a = findfirst("ufun(n) =", src)[1]; b = findfirst("relerr(a, b)", src)[1]
eval(Meta.parseall(src[a:b-1]))
n, nstep = 12, 7; c, dx, dt = 0.4, 1.0, 0.1
u0 = ufun(n); du0 = dufun(n); ubs = ubfun(n); dubs = dubfun(n)
J(u0, du0, c, dx, dt) = (r = cpu_primal(u0, du0, c, dx, dt, nstep); sum(ubs .* r[1]) + sum(dubs .* r[2]))
(uf, duf, ub, dub, cb, dxb, dtb) = cpu_adjoint(u0, du0, ubs, dubs, c, dx, dt, nstep)
h = 1e-6; fd(f, x) = (f(x + h) - f(x - h)) / (2h)
e_c  = abs(fd(x -> J(u0, du0, x, dx, dt), c) - cb) / abs(cb)
e_dx = abs(fd(x -> J(u0, du0, c, x, dt), dx) - dxb) / abs(dxb)
e_dt = abs(fd(x -> J(u0, du0, c, dx, x), dt) - dtb) / abs(dtb)
gu = [ (e = zeros(n); e[i] = h; (J(u0 .+ e, du0, c, dx, dt) - J(u0 .- e, du0, c, dx, dt)) / (2h)) for i in 1:n]
gd = [ (e = zeros(n); e[i] = h; (J(u0, du0 .+ e, c, dx, dt) - J(u0, du0 .- e, c, dx, dt)) / (2h)) for i in 1:n]
println("scalar grads rel err vs FD: c=", e_c, " dx=", e_dx, " dt=", e_dt)
println("grad wrt u0:  max abs err vs FD = ", maximum(abs.(gu .- ub)), "  (max |grad| = ", maximum(abs.(ub)), ")")
println("grad wrt du0: max abs err vs FD = ", maximum(abs.(gd .- dub)), "  (dub[1] = ", dub[1], ", dub[2:end] max = ", maximum(abs.(dub[2:end])), ")")
