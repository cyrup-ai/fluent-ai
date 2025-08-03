#!/usr/bin/env perl
use strict;
use warnings;
use File::Find;
use File::Slurp;

# Fix AsyncStream iterator issues in test files
my @test_dirs = ('tests/');

find(
    sub {
        return unless /\.rs$/;
        return if /^lib\.rs$/;
        
        my $file = $File::Find::name;
        my $content;
        eval { $content = read_file($file); };
        if ($@) {
            warn "Could not read $file: $@";
            return;
        }
        my $original = $content;
        
        # Fix pattern 1: .map(|r| r.expect("...")) on AsyncStream
        $content =~ s/\.process_chunk\([^)]+\)\s*\n?\s*\.map\(\|r\|\s*r\.expect\([^)]+\)\)/\.process_chunk($1)\.collect()/g;
        
        # Fix pattern 2: .filter_map(|r| r.ok()) on AsyncStream
        $content =~ s/\.process_chunk\([^)]+\)\s*\n?\s*\.filter_map\(\|r\|\s*r\.ok\(\)\)/\.process_chunk($1)\.collect()/g;
        
        # Fix pattern 3: Direct iteration over AsyncStream
        $content =~ s/for\s+result\s+in\s+([^.]+)\.process_chunk\(([^)]+)\)\s*\{/my \$results = $1.process_chunk($2).collect();\n        for result in results {/g;
        
        # Fix pattern 4: Remove Result pattern matching where AsyncStream returns direct values
        $content =~ s/if\s+let\s+Ok\(([^)]+)\)\s*=\s*result\s*\{/if true { let $1 = result;/g;
        $content =~ s/match\s+result\s*\{\s*Ok\(([^)]+)\)\s*=>\s*/let $1 = result; {/g;
        $content =~ s/Err\([^)]*\)\s*=>\s*[^,}]+[,}]/}/g;
        
        # Fix pattern 5: .as_ref().ok() where not needed
        $content =~ s/\.as_ref\(\)\.ok\(\)//g;
        
        if ($content ne $original) {
            write_file($file, $content);
            print "Fixed AsyncStream issues in: $file\n";
        }
    },
    @test_dirs
);

print "AsyncStream test fixes completed.\n";