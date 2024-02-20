tell application "Keynote"
    activate
    -- Make sure a presentation is open
    if not (exists front document) then error "Please open a presentation first."
    set thePresentation to front document
    set slideText to ""
    
    -- Loop through each slide in the presentation
    repeat with i from 1 to the count of slides of thePresentation
        set thisSlide to slide i of thePresentation
        set slideText to slideText & "Slide " & i & ":"
        
        -- Loop through each text item in the slide
        repeat with j from 1 to the count of text items of thisSlide
            set thisTextItem to text item j of thisSlide
            set theText to object text of thisTextItem
            set slideText to slideText & return & theText
        end repeat
        set slideText to slideText & return & return
    end repeat
end tell

-- Writing the extracted text to a file on the desktop
set desktopPath to (path to desktop folder as text) & "KeynoteText.txt"
set fileReference to open for access file desktopPath with write permission
write slideText to fileReference
close access fileReference

-- Notifying the user that the script has finished
display notification "Extracted text has been saved to KeynoteText.txt on your desktop." with title "Extraction Complete"