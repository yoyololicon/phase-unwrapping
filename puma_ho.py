import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def circshift(x, shifts):
    return np.roll(np.roll(x, shifts[1], axis=1), shifts[0], axis=0)


def clique_energy_ho(x, p):
    return np.abs(x) ** p


def energy_ho(kappa, psi, base, p, cliques, disc_bar):
    M, N = psi.shape
    cliquesm, cliquesn = cliques.shape
    maxdesl = np.abs(cliques).max()

    base_kappa = np.pad(kappa, ((maxdesl + 1,) * 2,) * 2, constant_values=0.)
    psi_base = np.pad(psi, ((maxdesl + 1,) * 2,) * 2, constant_values=0.)
    z = disc_bar.shape[2]
    base_disc_bar = np.pad(disc_bar, ((maxdesl + 1,) * 2,) * 2 + ((0, 0),), constant_values=0.)

    a = []
    for t in range(cliquesm):
        base_start = circshift(base, [-cliques[t, 0], -cliques[t, 1]]) * base
        base_end = circshift(base, [cliques[t, 0], cliques[t, 1]]) * base
        auxili = circshift(base_kappa, [cliques[t, 0], cliques[t, 1]])
        t_dkappa = base_kappa - auxili
        auxili2 = circshift(psi_base, [cliques[t, 0], cliques[t, 1]])
        dpsi = auxili2 - psi_base

        a.append(
            (2 * np.pi * t_dkappa - dpsi) *
            base * circshift(base, [cliques[t, 0], cliques[t, 1]]) * base_disc_bar[..., t]
        )

    a = np.stack(a, axis=2)
    return clique_energy_ho(a, p).sum()


def puma_ho(psi, p=2, cliques=((0, 1), (1, 0)), schedule=[1]):
    M, N = psi.shape
    kappa = np.zeros_like(psi)
    kappa_aux = kappa
    iter = 0
    erglist = []
    cliques = np.array(cliques)
    cliquesm, cliquesn = cliques.shape
    qualitymaps = np.zeros((*psi.shape, cliques.shape[0]))

    disc_bar = 1 - qualitymaps
    maxdesl = np.abs(cliques).max()

    base = np.pad(np.ones_like(psi), ((maxdesl + 1,) * 2,) * 2, constant_values=0.)

    for jump_size in schedule:
        erg_previous = energy_ho(kappa, psi, base, p, cliques, disc_bar)
        print(erg_previous)

        while True:
            iter += 1
            erglist.append(erg_previous)
            remain = []

            base_kappa = np.pad(kappa, ((maxdesl + 1,) * 2,) * 2, constant_values=0.)
            psi_base = np.pad(psi, ((maxdesl + 1,) * 2,) * 2, constant_values=0.)

            z = disc_bar.shape[2]
            base_disc_bar = np.pad(disc_bar, ((maxdesl + 1,) * 2,) * 2 + ((0, 0),), constant_values=0.)

            a, A, B, C, D = [], [], [], [], []
            source, sink = [], []
            base_start, base_end = [], []
            for t in range(cliquesm):
                base_start.append(circshift(base, [-cliques[t, 0], -cliques[t, 1]]) * base)
                base_end.append(circshift(base, [cliques[t, 0], cliques[t, 1]]) * base)
                auxili = circshift(base_kappa, [cliques[t, 0], cliques[t, 1]])
                t_dkappa = base_kappa - auxili
                auxili2 = circshift(psi_base, [cliques[t, 0], cliques[t, 1]])
                dpsi = auxili2 - psi_base

                a.append(
                    (2 * np.pi * t_dkappa - dpsi) *
                    base * circshift(base, [cliques[t, 0], cliques[t, 1]])
                )
                A.append(
                    clique_energy_ho(abs(a[-1]), p) * base *
                    circshift(base, [cliques[t, 0], cliques[t, 1]]) * base_disc_bar[:, :, t]
                )
                D.append(A[-1])
                C.append(
                    clique_energy_ho(abs(2 * np.pi * jump_size + a[-1]), p) * base *
                    circshift(base, [cliques[t, 0], cliques[t, 1]]) * base_disc_bar[:, :, t]
                )
                B.append(
                    clique_energy_ho(abs(-2 * np.pi * jump_size + a[-1]), p) * base *
                    circshift(base, [cliques[t, 0], cliques[t, 1]]) * base_disc_bar[:, :, t]
                )

                source.append(
                    circshift((C[-1] - A[-1]) * ((C[-1] - A[-1]) > 0),
                              [-cliques[t, 0], -cliques[t, 1]]) * base_start[-1]
                )
                sink.append(
                    circshift((A[-1] - C[-1]) * ((A[-1] - C[-1]) > 0),
                              [-cliques[t, 0], -cliques[t, 1]]) * base_start[-1]
                )
                source[-1] += ((D[-1] - C[-1]) * ((D[-1] - C[-1]) > 0)) * base_end[-1]
                sink[-1] += ((C[-1] - D[-1]) * ((C[-1] - D[-1]) > 0)) * base_end[-1]

            a = np.stack(a, axis=2)
            A = np.stack(A, axis=2)
            B = np.stack(B, axis=2)
            C = np.stack(C, axis=2)
            D = np.stack(D, axis=2)
            source = np.stack(source, axis=2)
            sink = np.stack(sink, axis=2)
            base_start = np.stack(base_start, axis=2)
            base_end = np.stack(base_end, axis=2)

            source = source[maxdesl + 1:-maxdesl - 1, maxdesl + 1:-maxdesl - 1]
            sink = sink[maxdesl + 1:-maxdesl - 1, maxdesl + 1:-maxdesl - 1]
            auxiliar1 = B + C - A - D
            auxiliar1 = auxiliar1[maxdesl + 1:-maxdesl - 1, maxdesl + 1:-maxdesl - 1]
            base_start = base_start[maxdesl + 1:-maxdesl - 1, maxdesl + 1:-maxdesl - 1]
            base_end = base_end[maxdesl + 1:-maxdesl - 1, maxdesl + 1:-maxdesl - 1]

            G = nx.DiGraph()
            for t in range(cliquesm):
                start = np.where(base_start[:, :, t].ravel() != 0)[0]
                endd = np.where(base_end[:, :, t].ravel() != 0)[0]
                print(start.shape, endd.shape)
                auxiliar2 = auxiliar1[:, :, t].ravel()
                plt.subplot(311)
                plt.imshow(base_start[..., t])
                plt.subplot(312)
                plt.imshow(base_end[..., t])
                plt.subplot(313)
                plt.imshow(auxiliar1[..., t])
                plt.show()
                print(start, endd, auxiliar2[endd])
                G.add_weighted_edges_from(zip(start, endd, np.maximum(0, auxiliar2[endd])), weight='capacity')

            source_final = source.sum(2)
            sink_final = sink.sum(2)
            s, t = M * N, M * N + 1
            G.add_weighted_edges_from(zip([s] * s, range(s), source_final.ravel()), weight='capacity')
            G.add_weighted_edges_from(zip(range(s), [t] * s, sink_final.ravel()), weight='capacity')


            print(len(G.edges.data()))
            print("caculating flow and min cut")
            cut_value, partition = nx.minimum_cut(G, s, t,
                                                  flow_func=nx.algorithms.flow.boykov_kolmogorov)
            reachable, _ = partition
            reachable.discard(s)
            print(len(reachable))

            kappa_aux = kappa.copy()
            if len(reachable):
                kappa_aux[np.unravel_index(list(reachable), (M, N))] += jump_size

            plt.imshow((kappa_aux - kappa).reshape(M, N))
            plt.show()

            erg_actual = energy_ho(kappa_aux, psi, base, p, cliques, disc_bar)
            print(erg_actual)


            if erg_actual < erg_previous:
                erg_previous = erg_actual
                kappa = kappa_aux
            else:
                unwph = 2 * np.pi * kappa + psi
                break

            plt.subplot(211)
            plt.imshow(kappa)
            plt.subplot(212)
            plt.imshow(2 * np.pi * kappa + psi)
            plt.show()


if __name__ == '__main__':
    from scipy.stats import multivariate_normal


    def insarpair_v2(power, cohe, phase, npower):
        M, N = phase.shape
        w1 = 1 / np.sqrt(2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(power);
        w2 = 1 / np.sqrt(2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(power);

        n1 = np.sqrt(npower / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
        n2 = np.sqrt(npower / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))

        x1 = cohe * w1 + np.sqrt(1 - cohe ** 2) * w2 + n1
        x2 = w1 * np.exp(-1j * phase) + n2
        return x1, x2


    shape = (100,) * 2
    height = 14 * np.pi
    mean = np.array(shape) * 0.5
    cov = np.array([10 ** 2, 10 ** 2, ])
    rv = multivariate_normal(mean, cov)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    output = rv.pdf(np.dstack((x, y)))
    output *= height / output.max()
    #output[:50, :50] = 0
    # output += np.random.randn(*output.shape) * 0.8
    co = 0.95
    x1, x2 = insarpair_v2(1, co, output, 0)
    psi = np.angle(x1 * np.conj(x2))
    # psi = wrap_func(output)

    puma_ho(psi, 2)
