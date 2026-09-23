"""
CSCI-561 Fall 2026 - Homework 1 Parth Gosar (9609181412)
3D Travelling Salesman Problem: genetic algorithm with 2-opt local search.

Reads input.txt and writes output.txt in the current directory.
"""

import math
import random
import time
import bisect
import heapq
from collections import deque

_clock = time.process_time
START_TIME = _clock()

TIME_LIMIT = 50.0

MATRIX_MAX = 2000

NEIGHBOR_K = 12
STALL_LIMIT = 150   # generations without progress before the GA is called converged
KICK_LIMIT = 1500   # fruitless perturbations before the search is called converged

_sqrt = math.sqrt



# I/O

def read_input(path="input.txt"):
    with open(path, "r") as f:
        tokens = f.read().split()
    n = int(tokens[0])
    cities = []
    idx = 1
    for _ in range(n):
        x = int(tokens[idx]); y = int(tokens[idx + 1]); z = int(tokens[idx + 2])
        cities.append((x, y, z))
        idx += 3
    return cities


def write_output(tour, cost, cities, path="output.txt"):
    lines = ["%.3f" % cost]
    for ci in tour:
        c = cities[ci]
        lines.append("%d %d %d" % (c[0], c[1], c[2]))
    c0 = cities[tour[0]]
    lines.append("%d %d %d" % (c0[0], c0[1], c0[2]))
    # newline="\n" keeps the file LF even if it is generated on Windows.
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")



# Distance

def euclid(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return _sqrt(dx * dx + dy * dy + dz * dz)


def build_matrix(cities):
    n = len(cities)
    mat = []
    for i in range(n):
        ci = cities[i]
        row = [0.0] * n
        for j in range(i + 1, n):
            row[j] = euclid(ci, cities[j])
        mat.append(row)
    for i in range(n):
        for j in range(i):
            mat[i][j] = mat[j][i]
    return mat


def make_dist(cities, mat):
    """Returning d(i, j) backed by the matrix when there is one."""
    if mat is not None:
        def dist(i, j):
            return mat[i][j]
    else:
        def dist(i, j):
            a = cities[i]; b = cities[j]
            dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
            return _sqrt(dx * dx + dy * dy + dz * dz)
    return dist


def build_neighbors_matrix(mat, k):
    n = len(mat)
    nbrs = []
    for i in range(n):
        row = mat[i]
        order = heapq.nsmallest(k + 1, range(n), key=row.__getitem__)
        nbrs.append([j for j in order if j != i][:k])
    return nbrs


def build_neighbors_grid(cities, k):
    """Approximate k-nearest lists via a uniform bucket grid.

    Exact lists would need the full matrix; for large N these are close enough to drive 2-opt and cost O(n) memory instead of O(n^2).
    """
    n = len(cities)
    minx = min(c[0] for c in cities); maxx = max(c[0] for c in cities)
    miny = min(c[1] for c in cities); maxy = max(c[1] for c in cities)
    minz = min(c[2] for c in cities); maxz = max(c[2] for c in cities)
    span = max(maxx - minx, maxy - miny, maxz - minz, 1)

    per_axis = max(1, int((n / 3.0) ** (1.0 / 3.0)))
    cw = float(span) / per_axis
    if cw <= 0.0:
        cw = 1.0

    grid = {}
    keys = []
    for i in range(n):
        c = cities[i]
        key = (int((c[0] - minx) / cw), int((c[1] - miny) / cw), int((c[2] - minz) / cw))
        keys.append(key)
        if key in grid:
            grid[key].append(i)
        else:
            grid[key] = [i]

    def collect(kx, ky, kz, r):
        out = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    b = grid.get((kx + dx, ky + dy, kz + dz))
                    if b:
                        out.extend(b)
        return out

    nbrs = []
    for i in range(n):
        kx, ky, kz = keys[i]
        r = 1
        cand = collect(kx, ky, kz, r)
        while len(cand) <= k and r < per_axis:
            r += 1
            cand = collect(kx, ky, kz, r)
        # one extra ring, so a slightly closer point just outside the box is not missed
        if r < per_axis:
            cand = collect(kx, ky, kz, r + 1)
        ci = cities[i]
        cand = [j for j in cand if j != i]
        nbrs.append(heapq.nsmallest(k, cand, key=lambda j: euclid(ci, cities[j])))
    return nbrs



# Tour cost


def tour_cost(tour, dist):
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist(tour[i], tour[i + 1])
    total += dist(tour[-1], tour[0])
    return total


# Construction heuristic

def nearest_neighbor_tour(start, n, dist):
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        best = -1
        best_d = float("inf")
        for j in range(n):
            if not visited[j]:
                d = dist(cur, j)
                if d < best_d:
                    best_d = d
                    best = j
        visited[best] = True
        tour.append(best)
        cur = best
    return tour

# 2-opt local search

def two_opt(tour, dist, nbrs, deadline):
    """
    First-improvement 2-opt driven by k-nearest candidate lists.

    Both tour edges at each city are examined, and the partner city may sit on either side of it in the array. 
    Restricting either of those makes the search blind on nearest-neighbor tours: 
    there the edge worth removing is the long jump back to an unvisited region, whose endpoints' near neighbours all lie behind it.
    """
    n = len(tour)
    if n < 5:
        return tour

    pos = [0] * n
    for i in range(n):
        pos[tour[i]] = i

    def reverse(i, j):
        """
        Reverse tour positions i..j, or the complementary arc when shorter.
        Both yield the same cycle, so always move at most n/2 elements.
        """
        inner = j - i + 1
        if inner * 2 <= n:
            li = i; ri = j
            while li < ri:
                tour[li], tour[ri] = tour[ri], tour[li]
                pos[tour[li]] = li
                pos[tour[ri]] = ri
                li += 1
                ri -= 1
        else:
            li = (j + 1) % n
            ri = (i - 1) % n
            for _ in range((n - inner) // 2):
                tour[li], tour[ri] = tour[ri], tour[li]
                pos[tour[li]] = li
                pos[tour[ri]] = ri
                li = (li + 1) % n
                ri = (ri - 1) % n

    queued = bytearray([1]) * n
    queue = deque(range(n))

    guard = 0
    while queue:
        a = queue.popleft()
        queued[a] = 0

        guard += 1
        if (guard & 127) == 0 and _clock() > deadline:
            return tour

        moved = False
        for step in (1, -1):
            ia = pos[a]
            b = tour[(ia + step) % n]
            d_ab = dist(a, b)
            for c in nbrs[a]:
                d_ac = dist(a, c)
                if d_ac >= d_ab:
                    break  
                if c == b:
                    continue
                ic = pos[c]
                e = tour[(ic + step) % n]
                if e == a:
                    continue
                # dropping (a,b) and (c,e), add (a,c) and (b,e)
                if d_ab + dist(c, e) - d_ac - dist(b, e) > 1e-9:
                    if step == 1:
                        p, q = ia, ic
                    else:
                        # the same move seen from the predecessors
                        p, q = (ia - 1) % n, (ic - 1) % n
                    if p > q:
                        p, q = q, p
                    reverse(p + 1, q)
                    # four now sit on edges they did not sit on before
                    for x in (a, b, c, e):
                        if not queued[x]:
                            queued[x] = 1
                            queue.append(x)
                    moved = True
                    break
            if moved:
                break
    return tour




# Or-opt search
def or_opt(tour, dist, nbrs, deadline):
    """Relocate chains of 1-3 cities, including wrap-around and reversed moves."""
    n = len(tour)
    if n < 5:
        return tour

    queued = bytearray([1]) * n
    queue = deque(range(n))
    checks = 0

    while queue:
        start_city = queue.popleft()
        queued[start_city] = 0

        checks += 1
        if (checks & 63) == 0 and _clock() > deadline:
            return tour

        pos = [0] * n
        for p, city in enumerate(tour):
            pos[city] = p

        i = pos[start_city]
        moved = False

        for length in (1, 2, 3):
            if length >= n - 1:
                continue

            chain_positions = [(i + t) % n for t in range(length)]
            chain = [tour[p] for p in chain_positions]
            chain_set = set(chain)

            first = chain[0]
            last = chain[-1]
            before = tour[(i - 1) % n]
            after = tour[(i + length) % n]

            remove_gain = (
                dist(before, first)
                + dist(last, after)
                - dist(before, after)
            )

            seen = set()
            for endpoint in (first, last):
                for city in nbrs[endpoint]:
                    if city in seen or city in chain_set:
                        continue
                    seen.add(city)

                    j = pos[city]
                    next_city = tour[(j + 1) % n]
                    if next_city in chain_set:
                        continue

                    old_edge = dist(city, next_city)
                    forward_cost = (
                        dist(city, first)
                        + dist(last, next_city)
                        - old_edge
                    )
                    reverse_cost = (
                        dist(city, last)
                        + dist(first, next_city)
                        - old_edge
                    )

                    forward_gain = remove_gain - forward_cost
                    reverse_gain = remove_gain - reverse_cost
                    if max(forward_gain, reverse_gain) <= 1e-9:
                        continue

                    # Build the remaining cycle in its current cyclic order.
                    remaining = [x for x in tour if x not in chain_set]
                    insert_at = remaining.index(city) + 1

                    if reverse_gain > forward_gain:
                        placed = list(reversed(chain))
                    else:
                        placed = chain

                    tour[:] = remaining[:insert_at] + placed + remaining[insert_at:]

                    for c in (before, after, city, next_city, first, last):
                        if not queued[c]:
                            queued[c] = 1
                            queue.append(c)

                    moved = True
                    break

                if moved:
                    break
            if moved:
                break

    return tour


def improve_tour(tour, dist, nbrs, deadline):
    """Combine 2-opt and Or-opt while respecting the same time budget."""
    if _clock() >= deadline:
        return tour

    tour = two_opt(tour, dist, nbrs, deadline)

    if _clock() < deadline:
        tour = or_opt(tour, dist, nbrs, deadline)

    # Or-opt can expose new improving 2-opt moves.
    if _clock() < deadline:
        tour = two_opt(tour, dist, nbrs, deadline)

    return tour


# Genetic algorithm components

def CreateInitialPopulation(size, cities, dist, nbrs, deadline):
    """A list of `size` paths: a few nearest-neighbour tours refined by 2-opt,
    padded out with random permutations for diversity."""
    n = len(cities)
    population = []
    base = list(range(n))

    n_seeds = min(size // 4, n, 8)
    starts = random.sample(range(n), n_seeds) if n_seeds > 0 else []
    for s in starts:
        if population and _clock() > deadline - 5.0:
            break
        t = nearest_neighbor_tour(s, n, dist)
        t = improve_tour(t, dist, nbrs, min(deadline, _clock() + 5.0))
        population.append(t)

    while len(population) < size:
        t = base[:]
        random.shuffle(t)
        population.append(t)
    return population


def CreateMatingPool(population, RankList):
    """
    Roulette-wheel selection.
    RankList: (index, fitness) tuples sorted by fitness descending.
    Returns one selected parent per population slot.
    """
    total = 0.0
    for _, fit in RankList:
        total += fit
    cumulative = []
    acc = 0.0
    if total <= 0.0:
        # every tour scored the same- select uniformly
        step = 1.0 / len(RankList)
        for i in range(len(RankList)):
            cumulative.append(step * (i + 1))
    else:
        for _, fit in RankList:
            acc += fit / total
            cumulative.append(acc)
    pool = []
    for _ in range(len(population)):
        r = random.random()
        k = bisect.bisect_left(cumulative, r)
        if k >= len(RankList):
            k = len(RankList) - 1
        pool.append(population[RankList[k][0]])
    return pool


def Crossover(Parent1, Parent2, Start_index, End_index):
    """
    Two-point crossover with TSP conflict repair.
    The child keeps Parent1in place; the remaining slots are filled from Parent2 in its own order, skipping cities already used.
    """
    n = len(Parent1)
    child = [-1] * n
    in_slice = set()
    for i in range(Start_index, End_index + 1):
        child[i] = Parent1[i]
        in_slice.add(Parent1[i])
    fill = [c for c in Parent2 if c not in in_slice]
    fi = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[fi]
            fi += 1
    return child


def double_bridge(tour):
    """Double-bridge kick; keep the perturbation local on large tours."""
    n = len(tour)
    if n < 8:
        return tour[:]

    if n >= 500:
        window = min(n - 1, max(24, n // 8))
        start = random.randrange(n)

        rotated = tour[start:] + tour[:start]
        a, b, c = sorted(random.sample(range(1, window), 3))

        kicked = (
            rotated[:a]
            + rotated[c:window]
            + rotated[b:c]
            + rotated[a:b]
            + rotated[window:]
        )

        shift = n - start
        return kicked[shift:] + kicked[:shift]

    a, b, c = sorted(random.sample(range(1, n), 3))
    return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]


def mutate(tour, p_inv=0.3, p_swap=0.1):
    n = len(tour)
    if n > 3 and random.random() < p_inv:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        seg = tour[i:j + 1]
        seg.reverse()
        tour[i:j + 1] = seg
    if n > 2 and random.random() < p_swap:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        tour[i], tour[j] = tour[j], tour[i]
    return tour



# Driver

def solve():
    random.seed(561)
    cities = read_input()
    n = len(cities)

    if n == 0:
        with open("output.txt", "w", newline="\n") as f:
            f.write("0.000\n")
        return
    if n == 1:
        write_output([0], 0.0, cities)
        return
    if n == 2:
        write_output([0, 1], 2.0 * euclid(cities[0], cities[1]), cities)
        return
    if n == 3:
        # every ordering of three cities is the same cycle
        tour = [0, 1, 2]
        c = euclid(cities[0], cities[1]) + euclid(cities[1], cities[2]) + euclid(cities[2], cities[0])
        write_output(tour, c, cities)
        return

    
    ident_cost = 0.0
    for i in range(n):
        ident_cost += euclid(cities[i], cities[(i + 1) % n])
    write_output(list(range(n)), ident_cost, cities)

    deadline = START_TIME + TIME_LIMIT

    mat = build_matrix(cities) if n <= MATRIX_MAX else None
    dist = make_dist(cities, mat)
    k = min(NEIGHBOR_K, n - 1)
    nbrs = build_neighbors_matrix(mat, k) if mat is not None else build_neighbors_grid(cities, k)

    if n <= 100:
        pop_size = 120
    elif n <= 300:
        pop_size = 80
    else:
        pop_size = 50

    population = CreateInitialPopulation(pop_size, cities, dist, nbrs, deadline)
    costs = [tour_cost(t, dist) for t in population]

    bi = min(range(pop_size), key=lambda i: costs[i])
    best_tour = population[bi][:]
    best_cost = costs[bi]
    write_output(best_tour, best_cost, cities)

    elite = 2
    stall = 0
    gen = 0
    try:
        while _clock() < deadline - 1.0 and stall < STALL_LIMIT:
            gen += 1
            RankList = sorted(
                ((i, 1.0 / (costs[i] + 1e-12)) for i in range(pop_size)),
                key=lambda t: t[1], reverse=True,
            )
            pool = CreateMatingPool(population, RankList)

            new_pop = []
            for r in range(elite):
                new_pop.append(population[RankList[r][0]][:])

            while len(new_pop) < pop_size:
                p1 = random.choice(pool)
                p2 = random.choice(pool)
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                new_pop.append(mutate(Crossover(p1, p2, i, j)))

            population = new_pop
            costs = [tour_cost(t, dist) for t in population]
            bi = min(range(pop_size), key=lambda i: costs[i])

            if gen % 3 == 0 or costs[bi] < best_cost:
                slice_end = min(deadline - 0.5, _clock() + 1.0)
                population[bi] = improve_tour(population[bi], dist, nbrs, slice_end)
                costs[bi] = tour_cost(population[bi], dist)

            if costs[bi] < best_cost - 1e-9:
                best_cost = costs[bi]
                best_tour = population[bi][:]
                stall = 0
            else:
                stall += 1
        
        dead_kicks = 0
        strength = 1
        while _clock() < deadline - 1.0:
            cand = best_tour[:]
            for _ in range(strength):
                cand = double_bridge(cand)
            cand = improve_tour(cand, dist, nbrs, min(deadline - 0.5, _clock() + 2.0))
            cand_cost = tour_cost(cand, dist)
            if cand_cost < best_cost - 1e-9:
                best_cost = cand_cost
                best_tour = cand
                dead_kicks = 0
                strength = 1
            else:
                dead_kicks += 1
                if dead_kicks >= KICK_LIMIT:
                    # Kicks at this strength have stopped paying off. 
                    # Escalating the perturbation rather than quit with budget still on the clock.
                    strength = min(strength + 1, 8)
                    dead_kicks = 0
    except Exception:
        pass  # keep whatever best_tour reached rather than dying with no answer

    best_tour = improve_tour(best_tour, dist, nbrs, deadline)
    best_cost = tour_cost(best_tour, dist)
    write_output(best_tour, best_cost, cities)


if __name__ == "__main__":
    solve()
