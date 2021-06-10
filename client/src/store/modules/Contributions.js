import { get } from '@/helpers/HTTPHelper'

const state = {
    contributions: [],
}

const mutations = {
    SET_CONTRIBUTIONS: function (state, contributions) {
        state.contributions = contributions
    },
    ADD_CONTRIBUTION: function (state, contribution) {
        state.contributions.push(contribution)
    },
    EDIT_CONTRIBUTION: function (state, payload) {
        state.contributions.splice(payload.index, 1, payload.contribution)
    },
    DELETE_CONTRIBUTION: function (state, index) {
        state.contributions.splice(index, 1)
    },
    RESET_CONTRIBUTIONS_STATE: function (state) {
        state.contributions = []
    },
}

const actions = {
    getContributions: async function (context) {
        const response = await get('/contributions')
        if (response.value) {
            context.commit('SET_CONTRIBUTIONS', response.data)
        }
    },
    addContribution: function (context, contribution) {
        context.commit('ADD_CONTRIBUTION', contribution)
    },
}

const getters = {
    contributions: (state) => state.contributions,
}

export default {
    namespaced: true,
    state,
    mutations,
    actions,
    getters,
}
