# --- Setup messages ---
puts "Checking bonds only for resid < 110. Cartoon will fade in as bonds form."

# --- Globals ---
global bonds cutoff cartoon_rep_index total_bonds min_opacity max_opacity show_fraction

proc update_protein_bonds { args } {
    global bonds cutoff cartoon_rep_index quicksurf_rep_index total_bonds min_opacity max_opacity show_fraction

    set mol [molinfo top]
    set frame [molinfo $mol get frame]
    set bonded_count 0

    set total_frames [molinfo $mol get numframes]
    set current_frame [molinfo $mol get frame]
    set frame_fraction [expr {double($current_frame) / ($total_frames - 1)}]

    foreach bond $bonds {
        if {[catch {set d [measure bond $bond molid $mol]}]} { continue }
        if {$d > $cutoff} {
            topo delbond [lindex $bond 0] [lindex $bond 1]
        } else {
            topo addbond [lindex $bond 0] [lindex $bond 1]
            incr bonded_count
        }
    }

    mol ssrecalc $mol

    # Fraction of bonds formed
    if {$total_bonds > 0} {
        set fraction [expr {double($bonded_count)/$total_bonds}]
    } else {
        set fraction 0.0
    }

    # Nonlinear fade: invisible until 40%, then ramp up
    if {$fraction < 0.4} {
        set opacity $max_opacity
    } else {
        set scaled [expr {($fraction - 0.4) / 0.6}]
        if {$scaled > 1} { set scaled 1 }
        set opacity [expr {round(100*($max_opacity - $scaled * ($max_opacity - $min_opacity)))/100.0}]
    }
    if {$opacity < 0.0} { set opacity 0.0 }
    if {$opacity > 1.0} { set opacity 1.0 }

    # --- Use pre-made materials to avoid per-frame edits ---
    set numsteps 10
    set step [expr {int(round($opacity * ($numsteps - 1)))}]
    set matname [format "CartoonFade%d" $step]

    if {[lsearch [material list] $matname] == -1} {
        material add $matname
        material change $matname opacity $opacity
    }

    mol modmaterial $cartoon_rep_index $mol $matname

    set show_fraction 0.999
    if {$bonded_count == 0 || $frame_fraction < $show_fraction} {
        mol showrep $mol $cartoon_rep_index off
	mol showrep $mol $quicksurf_rep_index off
    } else {
        mol showrep $mol $cartoon_rep_index on
	mol showrep $mol $quicksurf_rep_index on
    }

    # --- Check if final frame reached ---
    set total_frames [molinfo $mol get numframes]
    if {$frame == [expr {$total_frames - 1}]} {
        puts "Final frame reached — switching to Cartoon + Backbone (Licorice) + QuickSurf representations."

        # Remove existing reps
        set numreps [molinfo $mol get numreps]
        for {set i [expr {$numreps - 1}]} {$i >= 0} {incr i -1} {
            mol delrep $i $mol
        }

        # --- Rep 1: QuickSurf for the whole protein ---
        mol color ColorID 2
        mol representation QuickSurf 1.0 0.5 1.0 1.0
        mol selection "resid < 110"
        mol material Transparent
        mol addrep $mol
	set quicksurf_rep_index [expr {[molinfo $mol get numreps] - 1}]

        # --- Rep 2A: Cartoon for the whole protein ---
        mol color ColorID 10
        mol representation NewCartoon 0.3 20.0 4.1 0
        mol selection "chain A and resid < 110"
        mol material AOChalky
        mol addrep $mol
	set cartoon_rep_index_A [expr {[molinfo $mol get numreps] - 1}]

        # --- Rep 2B: Cartoon for the whole protein ---
        mol color ColorID 15
        mol representation NewCartoon 0.3 20.0 4.1 0
        mol selection "chain B and resid < 110"
        mol material AOChalky
        mol addrep $mol
	set cartoon_rep_index_B [expr {[molinfo $mol get numreps] - 1}]

        # --- Rep 2C: Cartoon for the whole protein ---
        mol color ColorID 23
        mol representation NewCartoon 0.3 20.0 4.1 0
        mol selection "chain C and resid < 110"
        mol material AOChalky
        mol addrep $mol
	set cartoon_rep_index_C [expr {[molinfo $mol get numreps] - 1}]

        # --- Rep 2D: Cartoon for the whole protein ---
        mol color ColorID 26
        mol representation NewCartoon 0.3 20.0 4.1 0
        mol selection "chain D and resid < 110"
        mol material AOChalky
        mol addrep $mol
	set cartoon_rep_index_D [expr {[molinfo $mol get numreps] - 1}]

        # --- Rep 2E: Cartoon for the whole protein ---
        mol color ColorID 11
        mol representation NewCartoon 0.3 20.0 4.1 0
        mol selection "chain E and resid < 110"
        mol material AOChalky
        mol addrep $mol
	set cartoon_rep_index_E [expr {[molinfo $mol get numreps] - 1}]

	# --- Rep 3: Licorice for backbone only ---
        #mol color Name
        #mol representation Licorice 0.3 22.0 22.0
        #mol selection "backbone and resid < 110"
        #mol material AOChalky
        #mol addrep $mol

	# --- Rep 4: Red spheres for resid 21 ---
        mol color ColorID 29
        mol representation VDW 1.0 12.0
        mol selection "resid 21"
        mol material AOChalky
        mol addrep $mol
    }
    
    puts "Frame $frame: $bonded_count / $total_bonds bonds → $matname (opacity $opacity)"
}

# --- Jump to final frame ---
set numframes [molinfo top get numframes]
animate goto $numframes

# --- Select subset ---
set sel [atomselect top "resid < 110"]
set indices [$sel get index]
set bonds_full [topo getbondlist]
set bonds {}

# keep only bonds with both atoms in our selection
foreach bond $bonds_full {
    if {([lsearch -exact $indices [lindex $bond 0]] != -1) && \
        ([lsearch -exact $indices [lindex $bond 1]] != -1)} {
        lappend bonds $bond
    }
}
set total_bonds [llength $bonds]
$sel delete

# --- Clear existing reps ---
set numreps [molinfo top get numreps]
for {set i 0} {$i < $numreps} {incr i} { mol delrep $i top }

# --- Rep 1: QuickSurf for the whole protein ---
mol color ColorID 15
mol representation QuickSurf 1.0 0.5 1.0 1.0
mol selection "resid < 110"
mol material Transparent
mol addrep top
set quicksurf_rep_index [expr {[molinfo top get numreps] - 1}]
mol showrep top $quicksurf_rep_index off

# --- Rep 1: Cartoon (start hidden) ---
mol color ColorID 15
mol representation NewCartoon 0.3 20.0 4.1 0
mol selection "resid < 110"
mol material AOChalky
mol addrep top
set cartoon_rep_index [expr {[molinfo top get numreps] - 1}]
mol showrep top $cartoon_rep_index off

# --- Rep 2: Licorice (real bonds will appear) ---
mol color Name
mol representation Licorice 0.3 22.0 22.0
mol selection "resid < 110"
mol material AOChalky
mol addrep top

# --- Rep 3: Red spheres for resid 21 ---
mol color ColorID 29
mol representation VDW 1.0 12.0
mol selection "resid 21"
mol material AOChalky
mol addrep top

# --- Parameters ---
set cutoff 2.5
set max_opacity 1.0   ;# fully transparent when no bonds
set min_opacity 0.05  ;# almost opaque when all bonds formed

# --- Start callback ---
trace variable vmd_frame([molinfo top]) w update_protein_bonds
update_protein_bonds 0

