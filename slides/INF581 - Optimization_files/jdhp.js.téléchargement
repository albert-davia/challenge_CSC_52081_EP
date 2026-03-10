// Ce code sera exécuté lorsque la page aura fini de charger
window.onload = function() {
    var currentLanguage = "en"; // Mettez la valeur de la langue actuelle ici

    var frNotes = document.querySelectorAll('.fr-notes');
    var enNotes = document.querySelectorAll('.en-notes');

    if (currentLanguage === "fr") {
        for(var i = 0; i < frNotes.length; i++) {
            frNotes[i].style.whiteSpace = 'pre-wrap';
        }

        // Hide English speaker notes
        for(var i = 0; i < enNotes.length; i++) {
            enNotes[i].style.display = 'none';
        }
    } else if (currentLanguage === "en") {
        for(var i = 0; i < enNotes.length; i++) {
            enNotes[i].style.whiteSpace = 'pre-wrap';
        }

        // Hide French speaker notes
        for(var i = 0; i < frNotes.length; i++) {
            frNotes[i].style.display = 'none';
        }
    }
}