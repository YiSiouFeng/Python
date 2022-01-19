# Writing Efficient Python Code


# 1. Foundations for efficiencies
In this chapter, you'll learn what it means to write efficient Python code. You'll explore Python's Standard Library, learn about NumPy arrays, and practice using some of Python's built-in tools. This chapter builds a foundation for the concepts covered ahead.

## A taste of things to come
* Print the list, new_list, that was created using a Non-Pythonic approach.
``` py
# Print the list created using the Non-Pythonic approach
i = 0
new_list= []
while i < len(names):
    if len(names[i]) >= 6:
        new_list.append(names[i])
    i += 1
print(new_list)
``` 

* A more Pythonic approach would loop over the contents of names, rather than using an index variable. Print better_list
```py
# Print the list created by looping over the contents of names
better_list = []
for name in names:
    if len(name) >= 6:
        better_list.append(name)
print(better_list)
```

* The best Pythonic way of doing this is by using list comprehension. Print best_list
```py
# Print the list created by using list comprehension
best_list = [name for name in names if len(name) >= 6]
print(best_list)
```

## Built-in practice: range()
* Create a range object that starts at zero and ends at five. Only use a stop argument.
* Convert the nums variable into a list called nums_list.
* Create a new list called nums_list2 that starts at one, ends at eleven, and increments by two by unpacking a range object using the star character (*).
```py
# Create a range object that goes from 0 to 5
nums = range(6)
print(type(nums))

# Convert nums to a list
nums_list = list(nums)
print(nums_list)

# Create a new list of odd numbers from 1 to 11 by unpacking a range object
nums_list2 = [*range(1,13,2)]
print(nums_list2)
```


## Built-in practice: enumerate()
* Instead of using for i in range(len(names)), update the for loop to use i as the index variable and name as the iterator variable and use enumerate().
* Rewrite the previous for loop using enumerate() and list comprehension to create a new list, indexed_names_comp.
* Create another list (indexed_names_unpack) by using the star character (*) to unpack the enumerate object created from using enumerate() on names. This time, start the index for enumerate() at one instead of zero.
```py
# Rewrite the for loop to use enumerate
indexed_names = []
for i,name in enumerate(names):
    index_name = (i,name)
    indexed_names.append(index_name) 
print(indexed_names)

# Rewrite the above for loop using list comprehension
indexed_names_comp = [(i,name) for i,name in enumerate(names)]
print(indexed_names_comp)

# Unpack an enumerate object with a starting index of one
indexed_names_unpack = [*enumerate(names, start =1)]
print(indexed_names_unpack)
```

## Built-in practice: map()
* Use map() and the method str.upper() to convert each name in the list names to uppercase. Save this to the variable names_map.
* Print the data type of names_map.
* Unpack the contents of names_map into a list called names_uppercase using the star character (*).
* Print names_uppercase and observe its contents.
```py
# Use map to apply str.upper to each element in names
names_map  = map(str.upper, names)

# Print the type of the names_map
print(type(names_map))

# Unpack names_map into a list
names_uppercase = [*names_map]

# Print the list created above
print(names_uppercase)
```

## Practice with NumPy arrays
* Print the second row of nums.
* Print the items of nums that are greater than six.
* Create nums_dbl that doubles each number in nums.
* Replace the third column in nums with a new column that adds 1 to each item in the original column.
```py
# Print second row of nums
print(nums[1,:])

# Print all elements of nums that are greater than six
print(nums[nums > 6])

# Double every element of nums
nums_dbl = nums* 2
print(nums_dbl)

# Replace the third column of nums
nums[:,2] = nums[:,2] + 1
print(nums)
```

## Bringing it all together: Festivus!
* Use range() to create a list of arrival times (10 through 50 incremented by 10). Create the list arrival_times by unpacking the range object.
```py
# Create a list of arrival times
arrival_times = [*range(10, 60, 10)]

print(arrival_times)
```
* You realize your clock is three minutes fast. Convert the arrival_times list into a numpy array (called arrival_times_np) and use NumPy broadcasting to subtract three minutes from each arrival time.
```py
# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

print(new_times)
```
* Use list comprehension with enumerate() to pair each guest in the names list to their updated arrival time in the new_times array. You'll need to use the index variable created from using enumerate() on new_times to index the names list.
```py
# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[name],time) for name,time in enumerate(new_times)]

print(guest_arrivals)
```
* Use list comprehension with enumerate() to pair each guest in the names list to their updated arrival time in the new_times array. You'll need to use the index variable created from using enumerate() on new_times to index the names list.
```py
# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]

# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest,guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')
```

# 2. Timing and profiling code
In this chapter, you will learn how to gather and compare runtimes between different coding approaches. You'll practice using the line_profiler and memory_profiler packages to profile your code base and spot bottlenecks. Then, you'll put your learnings to practice by replacing these bottlenecks with efficient Python code.

## Using %timeit: your turn!
* Use list comprehension and range() to create a list of integers from 0 to 50 called nums_list_comp.
* Use range() to create a list of integers from 0 to 50 and unpack its contents into a list called nums_unpack.
```py
# Create a list of integers (0-50) using list comprehension
nums_list_comp = [num for num in range(51)]
print(nums_list_comp)

# Create a list of integers (0-50) by unpacking range
nums_unpack = [*range(51)]
print(nums_unpack)
```

## Using %timeit: formal name or literal syntax
* Create an empty list called formal_list using the formal name (list()).
* Create an empty list called literal_list using the literal syntax ([]).
```py
# Create a list using the formal name
formal_list = list()
print(formal_list)

# Create a list using the literal syntax
literal_list = []
print(literal_list)
```
* Print out the type of formal_list and literal_list to show that both naming conventions create a list.
```py
# Create a list using the formal name
formal_list = list()
print(formal_list)

# Create a list using the literal syntax
literal_list = []
print(literal_list)

# Print out the type of formal_list
print(type(formal_list))

# Print out the type of literal_list
print(type(literal_list))
```


## Using %lprun: spot bottlenecks
* What percentage of time is spent on the new_hts list comprehension line of code relative to the total amount of time spent in the convert_units() function?
```py
%load_ext line_profiler
%lprun -f convert_units convert_units(heroes,hts,wts)
->14.8%
```

## Using %mprun: Hero BMI
* How much memory do the list comprehension lines of code consume in the calc_bmi_lists() function? (i.e., what is the total sum of the Increment column for these four lines of code?)
```py
%load_ext memory_profiler
from bmi_lists import calc_bmi_lists
%mprun -f calc_bmi_lists calc_bmi_lists(sample_indices,hts,wts)
```

## Bringing it all together: Star Wars profiling
* Use the get_publisher_heroes() function and the get_publisher_heroes_np() function to collect heroes from the Star Wars universe. The desired_publisher for Star Wars is 'George Lucas'
```py
# Use get_publisher_heroes() to gather Star Wars heroes
star_wars_heroes = get_publisher_heroes(heroes, publishers, 'George Lucas')

print(star_wars_heroes)
print(type(star_wars_heroes))

# Use get_publisher_heroes_np() to gather Star Wars heroes
star_wars_heroes_np = get_publisher_heroes_np(heroes, publishers, 'George Lucas')

print(star_wars_heroes_np)
print(type(star_wars_heroes_np))
```

# 3. Gaining efficiencies
This chapter covers more complex efficiency tips and tricks. You'll learn a few useful built-in modules for writing efficient code and practice using set theory. You'll then learn about looping patterns in Python and how to make them more efficient.

## *Let's Battle!!!*

## Combining Pokémon names and types
* Combine the names list and the primary_types list into a new list object (called names_type1).
```py
# Combine names and primary_types
names_type1 = [*zip(names, primary_types)]

print(*names_type1[:5], sep='\n')
```

* Combine names, primary_types, and secondary_types (in that order) using zip() and unpack the zip object into a new list.
```py
# Combine all three lists together
names_types = [*zip(names,primary_types,secondary_types)]

print(*names_types[:5], sep='\n')
```

* Use zip() to combine the first five items from the names list and the first three items from the primary_types list.
```py
# Combine five items from names and three items from primary_types
differing_lengths = [*zip(names[:5], primary_types[:3])]

print(*differing_lengths, sep='\n')
```
## Counting Pokémon from a sample
* Collect the count of each primary type from the sample.
* Collect the count of each generation from the sample.
* Use list comprehension to collect the first letter of each Pokémon in the names list. Save this as starting_letters.
* Collect the count of starting letters from the starting_letters list. Save this as starting_letters_count.
```py
# Collect the count of primary types
type_count = Counter(primary_types)
print(type_count, '\n')

# Collect the count of generations
gen_count = Counter(generations)
print(gen_count, '\n')

# Use list comprehension to get each Pokémon's starting letter
starting_letters = [name[0] for name in names]

# Collect the count of Pokémon for each starting_letter
starting_letters_count = Counter(starting_letters)
print(starting_letters_count)
```
## Combinations of Pokémon
* Import combinations from itertools.
* Create a combinations object called combos_obj that contains all possible pairs of Pokémon from the pokemon list. A pair has 2 Pokémon.
* Unpack combos_obj into a list called combos_2.
* Ash upgraded his Pokédex so that it can now store four Pokémon. Use combinations to collect all possible combinations of 4 different Pokémon. Save these combinations directly into a list called combos_4 using the star character (*).
```py
# Import combinations from itertools
from itertools import combinations

# Create a combination object with pairs of Pokémon
combos_obj = combinations(pokemon, 2)
print(type(combos_obj), '\n')

# Convert combos_obj to a list by unpacking
combos_2 = [*combos_obj]
print(combos_2, '\n')

# Collect all possible combinations of 4 Pokémon directly into a list
combos_4 = [*combinations(pokemon,4)]
print(combos_4)
```

## Comparing Pokédexes
* Convert both lists (ash_pokedex and misty_pokedex) to sets called ash_set and misty_set respectively.
* Find the Pokémon that both Ash and Misty have in common using a set method.
* Find the Pokémon that are within Ash's Pokédex but are not within Misty's Pokédex with a set method.
* Use a set method to find the Pokémon that are unique to either Ash or Misty (i.e., the Pokémon that exist in exactly one of the Pokédexes but not both).
```py
# Convert both lists to sets
ash_set = set(ash_pokedex)
misty_set = set(misty_pokedex)

# Find the Pokémon that exist in both sets
both = ash_set.intersection(misty_set)
print(both)

# Find the Pokémon that Ash has and Misty does not have
ash_only = ash_set.difference(misty_set)
print(ash_only)

# Find the Pokémon that are in only one set (not both)
unique_to_set = ash_set.symmetric_difference(misty_pokedex)
print(unique_to_set)
```
 
## Searching for Pokémon
* Convert Brock's Pokédex list (brock_pokedex) to a set called brock_pokedex_set.
* Check if 'Psyduck' is in Ash's Pokédex list (ash_pokedex) and if 'Psyduck' is in Brock's Pokédex set (brock_pokedex_set).
* Check if 'Machop' is in Ash's Pokédex list (ash_pokedex) and if 'Machop' is in Brock's Pokédex set (brock_pokedex_set).
```py
# Convert Brock's Pokédex to a set
brock_pokedex_set = set(brock_pokedex)
print(brock_pokedex_set)

# Check if Psyduck is in Ash's list and Brock's set
print('Psyduck' in ash_pokedex)
print('Psyduck' in brock_pokedex_set)

# Check if Machop is in Ash's list and Brock's set
print('Machop' in ash_pokedex)
print('Machop' in brock_pokedex_set)
```
=> Member testing using a set is faster than using a list in all four cases.

## Gathering unique Pokémon
* Use the provided function to collect the unique Pokémon in the names list. Save this as uniq_names_func.
* Use a set data type to collect the unique Pokémon in the names list. Save this as uniq_names_set.
```py
# Use find_unique_items() to collect unique Pokémon names
uniq_names_func = find_unique_items(names)
print(len(uniq_names_func))

# Convert the names list to a set to collect unique Pokémon names
uniq_names_set = set(uniq_names_func)
print(len(uniq_names_set))

# Check that both unique collections are equivalent
print(sorted(uniq_names_func) == sorted(uniq_names_set))
```
* Use the most efficient approach for gathering unique items to collect the unique Pokémon types (from the primary_types list) and Pokémon generations (from the generations list).
```py
# Use find_unique_items() to collect unique Pokémon names
uniq_names_func = find_unique_items(names)
print(len(uniq_names_func))

# Convert the names list to a set to collect unique Pokémon names
uniq_names_set = set(names)
print(len(uniq_names_set))

# Check that both unique collections are equivalent
print(sorted(uniq_names_func) == sorted(uniq_names_set))

# Use the best approach to collect unique primary types and generations
uniq_types = set(primary_types)
uniq_gens = set(generations)
print(uniq_types, uniq_gens, sep='\n') 
```
## Gathering Pokémon without a loop
* Eliminate the above for loop using list comprehension and the map() function:
* Use list comprehension to collect each Pokémon that belongs to generation 1 or generation 2. Save this as gen1_gen2_pokemon.
* Use the map() function to collect the number of letters in each Pokémon's name within the gen1_gen2_pokemon list. Save this map object as name_lengths_map.
* Combine gen1_gen2_pokemon and name_length_map into a list called gen1_gen2_name_lengths.
```py
# Collect Pokémon that belong to generation 1 or generation 2
gen1_gen2_pokemon = [name for name,gen in zip(poke_names, poke_gens) if gen < 3]

# Create a map object that stores the name lengths
name_lengths_map = map(len, gen1_gen2_pokemon)

# Combine gen1_gen2_pokemon and name_lengths_map into a list
gen1_gen2_name_lengths = [*zip(gen1_gen2_pokemon, name_lengths_map)]

print(gen1_gen2_name_lengths_loop[:5])
print(gen1_gen2_name_lengths[:5])
```

## Pokémon totals and averages without a loop
* Replace the above for loop using NumPy:
* Create a total stats array (total_stats_np) using the .sum() method and specifying the correct axis.
* Create an average stats array (avg_stats_np) using the .mean() method and specifying the correct axis.
* Create a final output list (poke_list_np) by combining the names list, the total_stats_np array, and the avg_stats_np array.
```py
# Create a total stats array
total_stats_np = stats.sum(axis=1)

# Create an average stats array
avg_stats_np = stats.mean(axis=1)

# Combine names, total_stats_np, and avg_stats_np into a list
poke_list_np = [*zip(names, total_stats_np, avg_stats_np)]

print(poke_list_np == poke_list, '\n')
print(poke_list_np[:3])
print(poke_list[:3], '\n')
top_3 = sorted(poke_list_np, key=lambda x: x[1], reverse=True)[:3]
print('3 strongest Pokémon:\n{}'.format(top_3))
```
## One-time calculation loop
* Import Counter from the collections module.
* Use Counter() to collect the count of each generation from the generations list. Save this as gen_counts.
* Write a better for loop that places a one-time calculation outside (above) the loop. Use the exact same syntax as the original for loop (simply copy and paste the one-time calculation above the loop).
```py
# Import Counter
from collections import Counter

# Collect the count of each generation
gen_counts = Counter(generations)

# Improve for loop by moving one calculation above the loop
total_count = len(generations)

for gen,count in gen_counts.items():
    gen_percent = round(count / total_count * 100, 2)
    print('generation {}: count = {:3} percentage = {}'
          .format(gen, count, gen_percent))
```
## Holistic conversion loop
* combinations from the itertools module has been loaded into your session. Use it to create a list called possible_pairs that contains all possible pairs of Pokémon types (each pair has 2 Pokémon types).
* Create an empty list called enumerated_tuples above the for loop.
* Within the for loop, append each enumerated_pair_tuple to the empty list you created in the above step.
* Use a built-in function to convert each tuple in enumerated_tuples to a list.
```py
# Collect all possible pairs using combinations()
possible_pairs = [*combinations(pokemon_types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

# Append each enumerated_pair_tuple to the empty list above
for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)
```
## Bringing it all together: Pokémon z-scores
* Use NumPy to eliminate the for loop used to create the z-scores.
* Then, combine the names, hps, and z_scores objects together into a list called poke_zscores2.
* Use list comprehension to replace the for loop used to collect Pokémon with the highest HPs based on their z-score.
```py
# Calculate the total HP avg and total HP standard deviation
hp_avg = hps.mean()
hp_std = hps.std()

# Use NumPy to eliminate the previous for loop
z_scores = (hps - hp_avg)/hp_std

# Combine names, hps, and z_scores
poke_zscores2 = [*zip(names, hps, z_scores)]
print(*poke_zscores2[:3], sep='\n')

# Use list comprehension with the same logic as the highest_hp_pokemon code block
highest_hp_pokemon2 = [(name,hp,zscore) for name,hp,zscore in poke_zscores2 if zscore > 2]
print(*highest_hp_pokemon2, sep='\n')
```

# 4. Basic pandas optimizations
This chapter offers a brief introduction on how to efficiently work with pandas DataFrames. You'll learn the various options you have for iterating over a DataFrame. Then, you'll learn how to efficiently apply functions to data stored in a DataFrame.

## Iterating with .iterrows()
* Use .iterrows() to loop over pit_df and print each row. Save the first item from .iterrows() as i and the second as row.
```py
# Iterate over pit_df and print each row
for i,row in pit_df.iterrows():
    print(row)
```
* Add two lines to the loop: one before print(row) to print each index variable and one after to print each row's type.
```py
# Iterate over pit_df and print each index variable and then each row
for i,row in pit_df.iterrows():
    print(i)
    print(row)
    print(type(row))
```
* Instead of using i and row in the for statement to store the output of .iterrows(), use one variable named row_tuple.
```py
# Use one variable instead of two to store the result of .iterrows()
for row_tuple in pit_df.iterrows():
    print(row_tuple)
```
* Add a line in the for loop to print the type of each row_tuple.
```py
# Print the row and type of each row
for row_tuple in pit_df.iterrows():
    print(row_tuple)
    print(type(row_tuple))
```
## Run differentials with .iterrows()
* Create an empty list called run_diffs that will be used to store the run differentials you will calculate
* Write a for loop that uses .iterrows() to loop over giants_df and collects each row's runs scored and runs allowed.
* Add a line to the for loop that uses the provided function to calculate each row's run differential.
```py
# Create an empty list to store run differentials
run_diffs = []

# Write a for loop and collect runs allowed and runs scored for each row
for i,row in giants_df.iterrows():
    runs_scored = row['RS']
    runs_allowed = row['RA']
    
    # Use the provided function to calculate run_diff for each row
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    # Append each run differential to the output list
    run_diffs.append(run_diff)

giants_df['RD'] = run_diffs
print(giants_df)
```
## Iterating with .itertuples()
* Use .itertuples() to loop over rangers_df and print each row.
* Loop over rangers_df with .itertuples() and save each row's Index, Year, and Wins (W) attribute as i, year, and wins.
* Now, loop over rangers_df and print these values only for those rows where the Rangers made the playoffs.
```py
# Loop over the DataFrame and print each row's Index, Year and Wins (W)
for row in rangers_df.itertuples():
  i = row.Index
  year = row.Year
  wins = row.W
  
  # Check if rangers made Playoffs (1 means yes; 0 means no)
  if row.Playoffs == 1:
    print(i, year,wins)
```
## Run differentials with .itertuples()
* Use .itertuples() to loop over yankees_df and grab each row's runs scored and runs allowed values.
* Now, calculate each row's run differential using calc_run_diff(). Be sure to append each row's run differential to run_diffs.
* Append a new column called 'RD' to the yankees_df DataFrame that contains the run differentials you calculated.
```py
run_diffs = []

# Loop over the DataFrame and calculate each row's run differential
for row in yankees_df.itertuples():
    
    runs_scored = row.RS
    runs_allowed = row.RA

    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    run_diffs.append(run_diff)

# Append new column
yankees_df['RD'] = run_diffs
print(yankees_df)
```
## Analyzing baseball stats with .apply()
* Apply sum() to each column of rays_df to collect the sum of each column. Be sure to specify the correct axis.

```py
# Gather sum of all columns
stat_totals = rays_df.apply(sum, axis=0)
print(stat_totals)
```
* Apply sum() to each row of rays_df, only looking at the 'RS' and 'RA' columns, and specify the correct axis.
```py
# Gather total runs scored in all games per year
total_runs_scored = rays_df[['RS', 'RA']].apply(sum, axis=1)
print(total_runs_scored)
```
* Use .apply() and a lambda function to apply text_playoffs() to each row's 'Playoffs' value of the rays_df DataFrame.
```py
# Convert numeric playoffs to text by applying text_playoffs()
textual_playoffs = rays_df.apply(lambda row: text_playoffs(row['Playoffs']), axis=1)
print(textual_playoffs)
```
## Settle a debate with .apply()
* Print the first five rows of the dbacks_df DataFrame to see what the data looks like.
* Create a pandas Series called win_percs by applying the calc_win_perc() function to each row of the DataFrame with a lambda function.
```py
# Display the first five rows of the DataFrame
print(dbacks_df.head())

# Create a win percentage Series 
win_percs = dbacks_df.apply(lambda row: calc_win_perc(row.W, row.G), axis=1)
print(win_percs, '\n')

# Append a new column to dbacks_df
dbacks_df['WP'] = win_percs
print(dbacks_df, '\n')

# Display dbacks_df where WP is greater than 0.50
print(dbacks_df[dbacks_df['WP'] >= 0.50])
```
## Replacing .iloc with underlying arrays
* Use the right method to collect the underlying 'W' and 'G' arrays of baseball_df and pass them directly into the calc_win_perc() function. Store the result as a variable called win_percs_np.
* Create a new column in baseball_df called 'WP' that contains the win percentages you just calculated.
* 
```py
# Use the W array and G array to calculate win percentages
win_percs_np = calc_win_perc(baseball_df['W'].values, baseball_df['G'].values)

# Append a new column to baseball_df that stores all win percentages
baseball_df['WP'] = win_percs_np

print(baseball_df.head())
```
## Bringing it all together: Predict win percentage
* Use a for loop and .itertuples() to predict the win percentage for each row of baseball_df with the predict_win_perc() function. Save each row's predicted win percentage as win_perc_pred and append each to the win_perc_preds_loop list.
* Apply predict_win_perc() to each row of the baseball_df DataFrame using a lambda function. Save the predicted win percentage as win_perc_preds_apply.
* Calculate the predicted win percentages by passing the underlying 'RS' and 'RA' arrays from baseball_df into predict_win_perc(). Save these predictions as win_perc_preds_np.
```py
win_perc_preds_loop = []

# Use a loop and .itertuples() to collect each row's predicted win percentage
for row in baseball_df.itertuples():
    runs_scored = row.RS
    runs_allowed = row.RA
    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)
    win_perc_preds_loop.append(win_perc_pred)

# Apply predict_win_perc to each row of the DataFrame
win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)

# Calculate the win percentage predictions using NumPy arrays
win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)
baseball_df['WP_preds'] = win_perc_preds_np
print(baseball_df.head())
```

*Completed by 2022/01/12*











