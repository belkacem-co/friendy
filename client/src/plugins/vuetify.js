import Vue from 'vue'
import Vuetify from 'vuetify/lib/framework'
import '@/styles/style.sass'

Vue.use(Vuetify)

export default new Vuetify({
    theme: {
        light: {
            primary: '#1976D2'
        }
    }
})
