using CSV, DataFrames, JuMP, HiGHS

df = CSV.read("ingredients.csv", DataFrame; comment="#") # from target's website

# Blank micros count as 0
col(name) = coalesce.(df[!, name], 0.0)

DAYS = 1

names = df.name
cost = col(:cost)
scale = col(:scale)
cal = col(:calories)
protein = col(:protein)
fat = col(:fats)
carbs = col(:carbs)
fiber = col(:fiber)
calcium = col(:ca)
iron = col(:fe)
magnesium = col(:mg)
potassium = col(:k)
sodium = col(:na)
zinc = col(:zn)
vit_c = col(:vit_c)
vit_d = col(:vit_d)
max_units = df.max

N = nrow(df)
M = maximum(max_units) * DAYS

# target macros
cost_range = (0, Inf) .* DAYS
CAL_MARGIN = 0 # how much you wanna save for condiments and such
cal_range = (1900 - CAL_MARGIN, 2434 - CAL_MARGIN) .* DAYS
protein_range = (150, Inf) .* DAYS
fat_range = (60, 109) .* DAYS
carbs_range = (100, Inf) .* DAYS
fiber_range = (25, Inf) .* DAYS
calcium_range = (1000, 2500) .* DAYS
iron_range = (8, 45) .* DAYS
magnesium_range = (400, Inf) .* DAYS
potassium_range = (3400, Inf) .* DAYS
sodium_range = (1500, 2300) .* DAYS
zinc_range = (11, 40) .* DAYS
vit_c_range = (90, 2000) .* DAYS
vit_d_range = (15, 100) .* DAYS

# Model and decision variables
model = Model(HiGHS.Optimizer)
@variable(model, x[1:N] >= 0, Int) # integer units of food `i` over the whole horizon
@variable(model, y[1:N], Bin)      # is food `i` used at all?

@constraint(model, [i = 1:N], x[i] <= max_units[i] * DAYS) # per-day cap * horizon
@constraint(model, [i = 1:N], x[i] <= M * y[i])            # big-M linking, x > 0 means y = 1
@constraint(model, [i = 1:N], x[i] >= y[i])                # y = 1 => x >= 1

CATEGORY_CAP = 1 # only allow this many overlapping categories. prevents the solver from suggesting 3 varieties of nugget
category = coalesce.(df.category, "other")
for c in unique(category)
    c == "supplement" && continue
    @constraint(model, sum(y[i] for i in 1:N if category[i] == c) <= CATEGORY_CAP)
end

# Compute totals
@expression(model, total_cost, sum(cost[i] * x[i] for i in 1:N))
@expression(model, total_cal, sum((scale[i] / 100.0) * cal[i] * x[i] for i in 1:N))
@expression(model, total_protein, sum((scale[i] / 100.0) * protein[i] * x[i] for i in 1:N))
@expression(model, total_fat, sum((scale[i] / 100.0) * fat[i] * x[i] for i in 1:N))
@expression(model, total_carbs, sum((scale[i] / 100.0) * carbs[i] * x[i] for i in 1:N))
@expression(model, total_fiber, sum((scale[i] / 100.0) * fiber[i] * x[i] for i in 1:N))
@expression(model, total_calcium, sum((scale[i] / 100.0) * calcium[i] * x[i] for i in 1:N))
@expression(model, total_iron, sum((scale[i] / 100.0) * iron[i] * x[i] for i in 1:N))
@expression(model, total_magnesium, sum((scale[i] / 100.0) * magnesium[i] * x[i] for i in 1:N))
@expression(model, total_potassium, sum((scale[i] / 100.0) * potassium[i] * x[i] for i in 1:N))
@expression(model, total_sodium, sum((scale[i] / 100.0) * sodium[i] * x[i] for i in 1:N))
@expression(model, total_zinc, sum((scale[i] / 100.0) * zinc[i] * x[i] for i in 1:N))
@expression(model, total_vit_c, sum((scale[i] / 100.0) * vit_c[i] * x[i] for i in 1:N))
@expression(model, total_vit_d, sum((scale[i] / 100.0) * vit_d[i] * x[i] for i in 1:N))

# Handle "closest to target" (absolute deviation)
@constraint(model, cost_range[1] <= total_cost <= cost_range[2])
@constraint(model, cal_range[1] <= total_cal <= cal_range[2])
@constraint(model, protein_range[1] <= total_protein <= protein_range[2])
@constraint(model, fat_range[1] <= total_fat <= fat_range[2])
@constraint(model, carbs_range[1] <= total_carbs <= carbs_range[2])
@constraint(model, fiber_range[1] <= total_fiber <= fiber_range[2])
@constraint(model, calcium_range[1] <= total_calcium <= calcium_range[2])
@constraint(model, iron_range[1] <= total_iron <= iron_range[2])
@constraint(model, magnesium_range[1] <= total_magnesium <= magnesium_range[2])
@constraint(model, potassium_range[1] <= total_potassium <= potassium_range[2])
@constraint(model, sodium_range[1] <= total_sodium <= sodium_range[2])
@constraint(model, zinc_range[1] <= total_zinc <= zinc_range[2])
@constraint(model, vit_c_range[1] <= total_vit_c <= vit_c_range[2])
@constraint(model, vit_d_range[1] <= total_vit_d <= vit_d_range[2])

# Objective function
VARIETY_COST = 10.0 # cost per unique ingredient. drives the optimizer to prefer smaller baskets and prevents it from suggesting 1 of a trillion different things
@objective(model, Min, total_cost + VARIETY_COST * sum(y))

# Solve and print results
optimize!(model)

if !is_solved_and_feasible(model)
    println("No solution: ", termination_status(model))
    exit(1)
end

println("Chosen ingredients:")
for i in 1:N
    units = Int(round(value(x[i]); digits=0))
    if units > 0
        println(units, "x ", rpad(names[i], 25))
    end
end

println("\nTotals:")
println("Cost: \$", round(value(total_cost); digits=2))
println("Calories: ", round(value(total_cal); digits=1))
println("Protein: ", round(value(total_protein); digits=1), "g")
println("Fat: ", round(value(total_fat); digits=1), "g")
println("Carbs: ", round(value(total_carbs); digits=1), "g")
println("Fiber: ", round(value(total_fiber); digits=1), "g")
println("Calcium: ", round(value(total_calcium); digits=1), "mg")
println("Iron: ", round(value(total_iron); digits=1), "mg")
println("Magnesium: ", round(value(total_magnesium); digits=1), "mg")
println("Potassium: ", round(value(total_potassium); digits=1), "mg")
println("Sodium: ", round(value(total_sodium); digits=1), "mg")
println("Zinc: ", round(value(total_zinc); digits=1), "mg")
println("Vit_C: ", round(value(total_vit_c); digits=1), "mg")
println("Vit_D: ", round(value(total_vit_d); digits=1), "µg")